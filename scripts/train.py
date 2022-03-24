import sys
sys.path.append(".")

import os
import logging
import random as py_random
from datetime import datetime

import numpy as np

import jax
from jax import random
from jax import numpy as jnp

import flax
from flax.training import checkpoints, train_state

import optax

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torch.utils.data import DataLoader

from nxcl.rich.logging import RichHandler, RichFileHandler
from nxcl.rich.progress import Progress

from models.cnn import CNN
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200


def compute_cross_entropy(*, logits, labels, num_classes):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    cross_entropy = -jnp.mean(jnp.sum(one_hot_labels * log_prob, axis=-1))
    return cross_entropy


def compute_accuracy(*, logits, labels):
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels) * 100.
    return accuracy


def get_train_step(model, use_batch_stats=False):
    @jax.jit
    def train_step(state, rngs, *, images, labels):
        def loss_fn(params):
            var = dict(params=params, batch_stats=state.batch_stats) if use_batch_stats else dict(params=params)
            outputs = state.apply_fn(var, images, rngs=rngs, train=True, mutable=(["batch_stats"] if use_batch_stats else False))
            if use_batch_stats:
                logits, new_state = outputs
            else:
                logits = outputs
            loss = compute_cross_entropy(logits=logits, labels=labels, num_classes=model.num_classes)
            accuracy = compute_accuracy(logits=logits, labels=labels)
            return loss, (accuracy, (new_state if use_batch_stats else None))

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (accuracy, new_state)), grads = grad_fn(state.params)

        if use_batch_stats:
            state = state.apply_gradients(grads=grads, batch_stats=new_state["batch_stats"])
        else:
            state = state.apply_gradients(grads=grads)
        return state, dict(loss=loss, accuracy=accuracy)
    return train_step


def get_valid_step(model, use_batch_stats=False):
    @jax.jit
    def valid_step(state, rngs, *, images, labels):
        var = dict(params=state.params, batch_stats=state.batch_stats) if use_batch_stats else dict(params=state.params)
        logits = state.apply_fn(var, images, rngs=rngs, train=False, mutable=False)
        loss = compute_cross_entropy(logits=logits, labels=labels, num_classes=model.num_classes)
        accuracy = compute_accuracy(logits=logits, labels=labels)
        return dict(loss=loss, accuracy=accuracy)
    return valid_step


def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)


def get_train_loader(dataset: str, batch_size: int):
    if dataset == "cifar10" or dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        CIFAR = CIFAR10 if dataset == "cifar10" else CIFAR100
        train_dataset = CIFAR(root="~/datasets", train=True, download=False, transform=train_transform)
    elif dataset == "imagenet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        train_dataset = ImageNet(root="~/datasets/ILSVRC2012", split="train", transform=train_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, collate_fn=numpy_collate)
    return train_loader


def get_valid_loader(dataset: str, batch_size: int):
    if dataset == "cifar10" or dataset == "cifar100":
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        CIFAR = CIFAR10 if dataset == "cifar10" else CIFAR100
        valid_dataset = CIFAR(root="~/datasets", train=False, download=False, transform=valid_transform)
    elif dataset == "imagenet":
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        valid_dataset = ImageNet(root="~/datasets/ILSVRC2012", split="val", transform=valid_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=4, collate_fn=numpy_collate)
    return valid_loader


def main(args, output_dir):
    logger = logging.getLogger(__name__)

    key = random.PRNGKey(args.seed)
    rngs = dict(params=key, dropout=key)

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

    num_classes = 10 if args.dataset == "cifar10" else 100 if args.dataset == "cifar100" else 1000
    model = models[args.model](num_classes=num_classes)

    if args.dataset == "cifar10" or args.dataset == "cifar100":
        dummy_input = jnp.ones((4, 32, 32, 3))
    elif args.dataset == "imagenet":
        dummy_input = jnp.ones((4, 224, 224, 3))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
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

    # Setup output directory
    exp_dir = os.path.join("outs", args.dataset, args.model, os.path.basename(output_dir))
    os.makedirs(os.path.dirname(exp_dir), exist_ok=True)
    os.symlink(os.path.join("..", "..", "_", os.path.basename(output_dir)), exp_dir)

    # Setup steps
    train_step = get_train_step(model, use_batch_stats=use_batch_stats)
    valid_step = get_valid_step(model, use_batch_stats=use_batch_stats)

    train_loader = get_train_loader(args.dataset, args.batch_size)
    valid_loader = get_valid_loader(args.dataset, args.batch_size)

    train_meter = AverageMeter("loss", "accuracy")
    valid_meter = AverageMeter("loss", "accuracy")

    # Train
    with Progress() as p:
        for i in p.trange(1, args.num_epochs + 1, description="Epoch"):

            train_meter.reset()
            for images, labels in p.track(train_loader, description="Train", remove=True):
                key, model_key = random.split(key)
                rngs = dict(dropout=model_key)
                state, train_metric = train_step(state, rngs, images=images, labels=labels)
                train_meter.update(train_metric, n=len(images))

            valid_meter.reset()
            for images, labels in p.track(valid_loader, description="Valid", remove=True):
                key, model_key = random.split(key)
                rngs = dict(dropout=model_key)
                valid_metric = valid_step(state, rngs, images=images, labels=labels)
                valid_meter.update(valid_metric, n=len(images))

            logger.info(
                f"Epoch {i:3d} / {args.num_epochs} | "
                f"Train Loss: {train_meter.loss:7.4f}  Acc: {train_meter.accuracy:6.2f} | "
                f"Valid Loss: {train_meter.loss:7.4f}  Acc: {train_meter.accuracy:6.2f}"
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

    logger = logging.getLogger(logger_name)
    logger.propagate = False
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


class AverageMeter:
    def __init__(self, *names):
        self.names = names
        self.sums = {k: 0 for k in names}
        self.cnts = {k: 0 for k in names}

    def reset(self):
        self.sums = {k: 0 for k in self.names}
        self.cnts = {k: 0 for k in self.names}

    # def update(self, values: Optional[dict] = None, n: int = 1, **kwargs):
    def update(self, values: dict = None, n: int = 1, **kwargs):
        if values is None:
            values = kwargs
        else:
            values = {**values, **kwargs}

        for k, v in values.items():
            self.sums[k] += v * n
            self.cnts[k] += n

    @property
    def value(self):
        return {k: self.sums[k] / self.cnts[k] for k in self.names}

    def __getattr__(self, name):
        if name in self.names:
            return self.sums[name] / self.cnts[name]
        else:
            raise AttributeError(f"{name} is not recorded metric")

    def __getitem__(self, name):
        if name in self.names:
            return self.sums[name] / self.cnts[name]
        else:
            raise KeyError(f"{name} is not recorded metric")


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
