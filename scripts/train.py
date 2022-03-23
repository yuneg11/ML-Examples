import sys
sys.path.append(".")

import os
import logging
import random as py_random
from datetime import datetime

import jax
from jax import random
from jax import numpy as jnp

import flax
from flax.training import checkpoints
from flax.training.train_state import TrainState

import optax

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from nxcl.logging import RichHandler, RichFileHandler

from lib.nets.cnn_v1 import CNN_V1


def compute_cross_entropy(*, logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    cross_entropy = -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))
    return cross_entropy


def compute_accuracy(*, logits, labels):
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels) * 100.
    return accuracy


def get_train_step(model):
    @jax.jit
    def train_step(state, rngs, *, images, labels):
        def loss_fn(params):
            logits = model.apply(params, images, rngs=rngs)
            loss = compute_cross_entropy(logits=logits, labels=labels)
            accuracy = compute_accuracy(logits=logits, labels=labels)
            return loss, accuracy
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, accuracy), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, dict(loss=loss, accuracy=accuracy)
    return train_step


def get_valid_step(model):
    @jax.jit
    def valid_step(state, rngs, *, images, labels):
        logits = model.apply(state.params, images, rngs=rngs)
        loss = compute_cross_entropy(logits=logits, labels=labels)
        accuracy = compute_accuracy(logits=logits, labels=labels)
        return dict(loss=loss, accuracy=accuracy)
    return valid_step


# def collate_fn(batch: torch.Tensor) -> jax.numpy.array:
#     images = jnp.asarray(torch.cat([image.unsqueeze(dim=0) for image, _ in batch], dim=0).numpy())
#     labels = jnp.asarray([label for _, label in batch])
#     return images, labels


def get_train_loader(batch_size: int):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        lambda x: x.permute(1, 2, 0),
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, #collate_fn=collate_fn,
    )
    return train_loader


def get_valid_loader(batch_size: int):
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        lambda x: x.permute(1, 2, 0),
    ])
    valid_dataset = CIFAR10(root='./data', train=False, download=False, transform=valid_transform)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, #collate_fn=collate_fn,
    )
    return valid_loader


def main(args, output_dir):
    logger = logging.getLogger(__name__)

    key = random.PRNGKey(args.seed)
    rngs = dict(params=key, dropout=key)

    # Create model
    if args.model == "cnn_v1":
        model = CNN_V1()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Initialize model and optimizer
    dummy_cifar10 = jnp.ones((4, 32, 32, 3))
    params = model.init(rngs, dummy_cifar10)
    # tx = optax.adam(learning_rate=args.learning_rate)
    tx = optax.sgd(learning_rate=args.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Setup output directory
    exp_dir = os.path.join("outs", args.model, os.path.basename(output_dir))
    os.makedirs(os.path.dirname(exp_dir), exist_ok=True)
    os.symlink(os.path.join("..", "_", os.path.basename(output_dir)), exp_dir)

    # Setup steps
    train_step = get_train_step(model)
    valid_step = get_valid_step(model)

    train_loader = get_train_loader(args.batch_size)
    valid_loader = get_valid_loader(args.batch_size)

    train_meter = AverageMeter("loss", "accuracy")
    valid_meter = AverageMeter("loss", "accuracy")

    # Train
    with Progress() as progress:
        epoch_task = progress.add_task("Epoch", total=args.num_epochs)

        for i in range(1, args.num_epochs + 1):
            progress.advance(epoch_task)
            key, model_key = random.split(key)

            train_task = progress.add_task("Train", total=len(train_loader))
            train_meter.reset()
            for images, labels in train_loader:
                images, labels = jnp.asarray(images.numpy()), jnp.asarray(labels.numpy())
                progress.advance(train_task)
                state, train_metric = train_step(state, dict(dropout=model_key), images=images, labels=labels)
                train_meter.update(train_metric, n=len(images))
            progress.remove_task(train_task)

            valid_task = progress.add_task("Valid", total=len(valid_loader))
            valid_meter.reset()
            for images, labels in valid_loader:
                images, labels = jnp.asarray(images.numpy()), jnp.asarray(labels.numpy())
                progress.advance(valid_task)
                valid_metric = valid_step(state, dict(dropout=model_key), images=images, labels=labels)
                valid_meter.update(valid_metric, n=len(images))
            progress.remove_task(valid_task)

            train_metric = train_meter.value
            valid_metric = valid_meter.value

            logger.info(
                f"Epoch {i:3d} / {args.num_epochs} | "
                f"Train Loss: {train_metric['loss']:7.4f}  Acc: {train_metric['accuracy']:6.2f} | "
                f"Valid Loss: {valid_metric['loss']:7.4f}  Acc: {valid_metric['accuracy']:6.2f}"
            )

            if i % args.save_every == 0:
                checkpoints.save_checkpoint(output_dir, state, i, keep=3)

    logger.info("Finished")


def get_experiment_name(random_code: str = None) -> str:
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    if random_code is None:
        random_code = "".join(py_random.choices("abcdefghikmnopqrstuvwxyz", k=4))
    return  now + "-" + random_code


def setup_logger(logger_name: str, output_dir: str):
    LOG_SHORT_FORMAT = "[%(asctime)s] %(message)s"
    LOG_LONG_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
    LOG_DATE_SHORT_FORMAT = "%H:%M:%S"
    LOG_DATE_LONG_FORMAT = "%Y-%m-%d %H:%M:%S"

    logging.getLogger().addHandler(logging.NullHandler())
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    stream_handler = RichHandler(tracebacks_suppress=[jax, flax, torch])
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(fmt=LOG_SHORT_FORMAT, datefmt=LOG_DATE_SHORT_FORMAT))
    logger.addHandler(stream_handler)

    debug_file_handler = RichFileHandler(os.path.join(output_dir, "debug.log"), mode="w")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(logging.Formatter(fmt=LOG_LONG_FORMAT, datefmt=LOG_DATE_LONG_FORMAT))
    logger.addHandler(debug_file_handler)

    info_file_handler = RichFileHandler(os.path.join(output_dir, "info.log"), mode="w", tracebacks_suppress=[jax, flax, torch])
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(logging.Formatter(fmt=LOG_SHORT_FORMAT, datefmt=LOG_DATE_LONG_FORMAT))
    logger.addHandler(info_file_handler)

    return logger


def Progress(*args, **kwargs):
    from rich import progress
    return progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
        progress.SpinnerColumn(),
        *args,
        **kwargs,
    )


class AverageMeter:
    def __init__(self, *args):
        self.names = args
        self.sums = {k: 0 for k in args}
        self.cnts = {k: 0 for k in args}

    def reset(self):
        self.sums = {k: 0 for k in self.names}
        self.cnts = {k: 0 for k in self.names}

    def update(self, values: dict, n: int = 1):
        for k in self.names:
            self.sums[k] += values[k] * n
            self.cnts[k] += n

    @property
    def value(self):
        return {k: self.sums[k] / self.cnts[k] for k in self.names}


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--model",   type=str,   required=True)
    parser.add_argument("-s", "--seed",    type=int,   default=0)
    parser.add_argument("--num-epochs",    type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=512)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--valid-every",   type=int,   default=100)
    parser.add_argument("--save-every",    type=int,   default=10)
    args = parser.parse_args()

    # Logger
    log_name = get_experiment_name()

    output_dir = os.path.join("outs", "_", log_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(__name__, output_dir)
    logger.debug("python " + " ".join(sys.argv))

    args_str = "\n  Arguments:"
    for k, v in vars(args).items():
        args_str += f"\n    {k:<15}: {v}"
    logger.info(args_str + "\n")

    try:
        main(args, output_dir)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception(e)
