import datetime
import os
import os.path
import platform
import gc
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() and platform.system() != "Windows" else "gloo",
        rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    distributed = world_size > 1
    try:
        if distributed:
            setup(rank, world_size, port)
        _run(rank, world_size, cfg, distributed=distributed)
    finally:
        if distributed:
            cleanup()


def _run(rank, world_size, cfg, distributed=True):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD(cfg).to(device)
    ddp_device_ids = [rank] if device.type == "cuda" else None
    if distributed:
        score_model = DDP(score_model, device_ids=ddp_device_ids, static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    if distributed:
        noise = DDP(noise, device_ids=ddp_device_ids, static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    
    # load in tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg, distributed=distributed)

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)


    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']


        if cfg.data.train != "text8":
            batch = next(train_iter)['input_ids'].to(device)
        else:
            batch = next(train_iter).to(device)
        loss = train_step_fn(state, batch)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            completed_step = state['step']
            if completed_step % cfg.training.log_freq == 0:
                if distributed:
                    dist.all_reduce(loss)
                    loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (completed_step, loss.item()))
            
            if completed_step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if completed_step % cfg.training.eval_freq == 0:
                if cfg.data.valid != "text8":
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                else:
                    eval_batch = next(train_iter).to(device)
                eval_loss = eval_step_fn(state, eval_batch)

                if distributed:
                    dist.all_reduce(eval_loss)
                    eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (completed_step, eval_loss.item()))

            if (completed_step > 0 and completed_step % cfg.training.snapshot_freq == 0) or completed_step == num_train_steps:
                # Save the checkpoint.
                save_step = completed_step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {completed_step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(completed_step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    ema.restore(score_model.parameters())

                    sentences = tokenizer.batch_decode(sample)
                    
                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("============================================================================================\n")

                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            batches = sample.shape[0] // cfg.eval.perplexity_batch_size
                            total_perplexity = 0
                            for i in range(batches):
                                s = sample[i * cfg.eval.perplexity_batch_size:(i + 1) * cfg.eval.perplexity_batch_size]
                                loss, logits = eval_model(s, labels=s)[:2]
                                logits = logits.transpose(-1, -2)
                                perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                                total_perplexity += perplexity
                            total_perplexity /= batches
                            if distributed:
                                dist.all_reduce(total_perplexity)
                                total_perplexity /= world_size
                            mprint(f"Generative Perplexity at step: {completed_step}. Perplexity: {total_perplexity:.3f}.")

                            del eval_model, logits, loss

                    if distributed:
                        dist.barrier()
