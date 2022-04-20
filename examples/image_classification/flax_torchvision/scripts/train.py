import sys
sys.path.append(".")

import os
import logging
from functools import partial

import numpy as np

import jax
from jax import lax
from jax import random
from jax import numpy as jnp

import flax
from flax import jax_utils
from flax.training import checkpoints, train_state

import optax

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageNet
from torch.utils.data import DataLoader

from nxml.jax.nn import functional as F
from nxcl.experimental import utils
from nxcl.rich import Progress

from models.cnn import CNN
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200


def get_train_loader(dataset: str, batch_size: int):
    if dataset == "cifar10" or dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.202, 0.199, 0.201)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        CIFAR = CIFAR10 if dataset == "cifar10" else CIFAR100
        train_dataset = CIFAR(root="~/datasets", train=True, download=False, transform=train_transform)
        num_workers = 4

    elif dataset == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.196, 0.198, 0.199)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        train_dataset = SVHN(root="~/datasets", split="train", download=False, transform=train_transform)
        num_workers = 4

    elif dataset == "imagenet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        train_dataset = ImageNet(root="~/datasets/ILSVRC2012", split="train", transform=train_transform)
        num_workers = 32

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, collate_fn=numpy_collate,
    )
    return train_loader


def get_valid_loader(dataset: str, batch_size: int):
    if dataset == "cifar10" or dataset == "cifar100":
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.202, 0.199, 0.201)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        CIFAR = CIFAR10 if dataset == "cifar10" else CIFAR100
        valid_dataset = CIFAR(root="~/datasets", train=False, download=False, transform=valid_transform)
        num_workers = 4

    elif dataset == "svhn":
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.196, 0.198, 0.199)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        valid_dataset = SVHN(root="~/datasets", split="test", download=False, transform=valid_transform)
        num_workers = 4

    elif dataset == "imagenet":
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        valid_dataset = ImageNet(root="~/datasets/ILSVRC2012", split="val", transform=valid_transform)
        num_workers = 32

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    valid_loader = DataLoader(
        valid_dataset, batch_size, shuffle=False, #drop_last=True,
        num_workers=num_workers, collate_fn=numpy_collate,
    )
    return valid_loader


@jax.jit
def sync_metric(metric):
    return jax.tree_util.tree_map(lambda x: jnp.mean(x), metric)


@partial(jax.pmap, axis_name="x")
def cross_replica_mean(state):
    return lax.pmean(state, axis_name="x")


def sync_batch_stats(state):
    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))
    return state


def get_train_step(use_batch_stats=False):
    @partial(jax.pmap, axis_name="batch")
    def _train_step(state, rngs, *, images, labels):
        def loss_fn(params):
            var = dict(params=params, batch_stats=state.batch_stats) if use_batch_stats else dict(params=params)
            outputs = state.apply_fn(var, images, rngs=rngs, train=True, mutable=(["batch_stats"] if use_batch_stats else False))
            if use_batch_stats:
                logits, new_state = outputs
            else:
                logits = outputs
            loss = F.cross_entropy(input=logits, target=labels)
            accuracy = F.accuracy(input=logits, target=labels)
            return loss, (accuracy, (new_state if use_batch_stats else None))

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (accuracy, new_state)), grads = grad_fn(state.params)
        grads = lax.pmean(grads, axis_name="batch")

        if use_batch_stats:
            state = state.apply_gradients(grads=grads, batch_stats=new_state["batch_stats"])
        else:
            state = state.apply_gradients(grads=grads)
        return state, dict(loss=loss, accuracy=accuracy)

    def train_step(state, rngs, *, images, labels):
        state, metric = _train_step(state, rngs, images=images, labels=labels)
        return state, sync_metric(metric)

    return train_step


def get_valid_step(use_batch_stats=False):
    @partial(jax.pmap, axis_name="batch")
    def _valid_step(state, rngs, *, images, labels):
        var = dict(params=state.params, batch_stats=state.batch_stats) if use_batch_stats else dict(params=state.params)
        logits = state.apply_fn(var, images, rngs=rngs, train=False, mutable=False)
        loss = F.cross_entropy(input=logits, target=labels)
        accuracy = F.accuracy(input=logits, target=labels)
        return dict(loss=loss, accuracy=accuracy)

    def valid_step(state, rngs, *, images, labels):
        metric = _valid_step(state, rngs, images=images, labels=labels)
        return sync_metric(metric)

    return valid_step


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def shard_batch(*arrs, n):
    new_arrs = []
    for arr in arrs:
        if arr.shape[0] % n != 0:
            raise ValueError(f"Batch size ({arr.shape[0]}) must be divisible by number of shards ({n})")
        new_arrs.append(np.reshape(arr, (n, arr.shape[0] // n, *arr.shape[1:])))
    return new_arrs


def main(args, output_dir):
    logger = logging.getLogger(__name__)

    key = random.PRNGKey(args.seed)
    rngs = dict(params=key, dropout=key)

    if args.dataset == "cifar10":
        dummy_input = jnp.ones((4, 32, 32, 3))
        num_classes = 10
    elif args.dataset == "cifar100":
        dummy_input = jnp.ones((4, 32, 32, 3))
        num_classes = 100
    elif args.dataset == "svhn":
        dummy_input = jnp.ones((4, 32, 32, 3))
        num_classes = 10
    elif args.dataset == "imagenet":
        dummy_input = jnp.ones((4, 224, 224, 3))
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create model
    models = {
        "cnn": CNN,
        "resnet18":  ResNet18,
        "resnet34":  ResNet34,
        "resnet50":  ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
        "resnet200": ResNet200,
    }

    if args.model not in models:
        raise ValueError(f"Unknown model: {args.model}")

    model = models[args.model](num_classes=num_classes)
    w = model.init(rngs, dummy_input)

    # Initialize model and optimizer
    if args.optimizer == "adam":
        tx = optax.adam(learning_rate=args.learning_rate)
    elif args.optimizer == "sgd":
        tx = optax.sgd(learning_rate=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if "batch_stats" in w:
        use_batch_stats = True
        class TrainState(train_state.TrainState):
            batch_stats: jnp.DeviceArray = None
        state = TrainState.create(apply_fn=model.apply, tx=tx, params=w["params"], batch_stats=w["batch_stats"])
    else:
        use_batch_stats = False
        class TrainState(train_state.TrainState):
            pass
        state = TrainState.create(apply_fn=model.apply, tx=tx, params=w["params"])

    state = jax_utils.replicate(state)
    num_devices = len(jax.local_devices())

    # Setup output directory
    utils.link_output_dir(output_dir, subnames=(args.dataset, args.model))

    # Setup steps
    train_step = get_train_step(use_batch_stats=use_batch_stats)
    valid_step = get_valid_step(use_batch_stats=use_batch_stats)

    train_loader = get_train_loader(args.dataset, args.batch_size)
    valid_loader = get_valid_loader(args.dataset, args.batch_size)

    train_meter = utils.AverageMeter("loss", "accuracy")
    valid_meter = utils.AverageMeter("loss", "accuracy")

    # Train
    with Progress() as p:
        for i in p.trange(1, args.num_epochs + 1, description="Epoch"):

            train_meter.reset()
            for j, (images, labels) in enumerate(p.track(train_loader, description="Train", remove=True)):
                key, model_key = random.split(key)
                rngs = jax_utils.replicate(dict(dropout=model_key))
                images, labels = shard_batch(images, labels, n=num_devices)
                state, train_metric = train_step(state, rngs, images=images, labels=labels)
                train_meter.update(train_metric, n=len(images))

            if use_batch_stats:
                state = sync_batch_stats(state)

            valid_meter.reset()
            for j, (images, labels) in enumerate(p.track(valid_loader, description="Valid", remove=True)):
                key, model_key = random.split(key)
                rngs = jax_utils.replicate(dict(dropout=model_key))
                images, labels = shard_batch(images, labels, n=num_devices)
                valid_metric = valid_step(state, rngs, images=images, labels=labels)
                valid_meter.update(valid_metric, n=len(images))

            logger.info(
                f"Epoch {i:3d} / {args.num_epochs} | "
                f"Train Loss: {train_meter.loss:7.4f}  Acc: {train_meter.accuracy * 100:6.2f} | "
                f"Valid Loss: {valid_meter.loss:7.4f}  Acc: {valid_meter.accuracy * 100:6.2f}"
            )

            if i % args.save_every == 0 and jax.process_index() == 0:
                _state = jax_utils.unreplicate(state)
                checkpoints.save_checkpoint(output_dir, _state, i, keep=3)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    logger.info("Finished")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m",   "--model",         type=str,   default="cnn")
    parser.add_argument("-d",   "--dataset",       type=str,   default="cifar10")
    parser.add_argument("-opt", "--optimizer",     type=str,   default="sgd")
    parser.add_argument("-e",   "--num-epochs",    type=int,   default=100)
    parser.add_argument("-bs",  "--batch-size",    type=int,   default=512)
    parser.add_argument("-lr",  "--learning-rate", type=float, default=0.1)
    parser.add_argument("-s",   "--seed",          type=int,   default=0)
    parser.add_argument("--valid-every",           type=int,   default=100)
    parser.add_argument("--save-every",            type=int,   default=10)
    args = parser.parse_args()

    # Logger
    log_name = utils.get_experiment_name()

    output_dir = os.path.join("outs", "_", log_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = utils.setup_logger(__name__, output_dir, suppress=[jax, flax, torch])
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
