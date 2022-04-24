import sys
sys.path.append(".")

import os
import logging
from functools import partial

import jax
from jax import lax
from jax import random
from jax import numpy as jnp

import flax
from flax import jax_utils
from flax.training import checkpoints, train_state

import optax

import numpy as np
import torch
import random as pyrandom

from nxml.jax.nn import functional as F
from nxcl.config import load_config, save_config, add_config_arguments, ConfigDict
from nxcl.rich import Progress
from nxcl.experimental import utils

from lib.data import get_dataset_spec, get_train_loader, get_valid_loader
from lib.models import CNN, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200


@jax.jit
def sync_metric(metric):
    return jax.tree_util.tree_map(lambda x: jnp.mean(x), metric)


@partial(jax.pmap, axis_name="replica")
def sync_replica_mean(state):
    return lax.pmean(state, axis_name="replica")


def sync_batch_stats(state):
    state = state.replace(batch_stats=sync_replica_mean(state.batch_stats))
    return state


def get_train_step(use_batch_stats=False):
    @partial(jax.pmap, axis_name="batch")
    def _train_step(state, rngs, *, images, labels):
        def loss_fn(params):
            if use_batch_stats:
                var = dict(params=params, batch_stats=state.batch_stats)
            else:
                var = dict(params=params)

            outputs = state.apply_fn(
                var, images, rngs=rngs, train=True,
                mutable=(["batch_stats"] if use_batch_stats else False),
            )

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
        if use_batch_stats:
            var = dict(params=state.params, batch_stats=state.batch_stats)
        else:
            var = dict(params=state.params)

        logits = state.apply_fn(var, images, rngs=rngs, train=False, mutable=False)
        loss = F.cross_entropy(input=logits, target=labels)
        accuracy = F.accuracy(input=logits, target=labels)
        return dict(loss=loss, accuracy=accuracy)

    def valid_step(state, rngs, *, images, labels):
        metric = _valid_step(state, rngs, images=images, labels=labels)
        return sync_metric(metric)

    return valid_step


def main(config, output_dir):
    num_devices = jax.local_device_count()

    # Logging
    logger = logging.getLogger(__name__)

    # Random seed
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    pyrandom.seed(config.train.seed)
    os.environ["PYTHONHASHSEED"] = str(config.train.seed)

    key = random.PRNGKey(config.train.seed)
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

    if config.model.name not in models:
        raise ValueError(f"Unknown model: {config.model.name}")

    spec = get_dataset_spec(config.dataset.name)

    model = models[config.model.name](num_classes=spec.num_classes)
    w = model.init(rngs, jnp.zeros((num_devices, *spec.image_shape)))

    # Initialize model and optimizer
    if config.optimizer.name == "sgd":
        tx = optax.sgd(learning_rate=config.optimizer.learning_rate)
    elif config.optimizer.name == "adam":
        tx = optax.adam(learning_rate=config.optimizer.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.name}")

    if "batch_stats" in w:
        use_batch_stats = True
        class TrainState(train_state.TrainState):
            batch_stats: jnp.DeviceArray = None
        state = TrainState.create(
            apply_fn=model.apply, tx=tx, params=w["params"], batch_stats=w["batch_stats"],
        )
    else:
        use_batch_stats = False
        class TrainState(train_state.TrainState):
            pass
        state = TrainState.create(apply_fn=model.apply, tx=tx, params=w["params"])

    state = jax_utils.replicate(state)

    # Setup output directory
    utils.link_output_dir(output_dir, subnames=(config.dataset.name, config.model.name))

    # Setup steps
    train_step = get_train_step(use_batch_stats=use_batch_stats)
    valid_step = get_valid_step(use_batch_stats=use_batch_stats)

    train_loader = get_train_loader(
        config.dataset.name, batch_size=config.dataset.batch_size,
        prefetch=config.dataset.prefetch, num_shards=num_devices,
    )
    valid_loader = get_valid_loader(
        config.dataset.name, batch_size=config.dataset.batch_size,
        prefetch=config.dataset.prefetch, num_shards=num_devices,
    )

    train_meter = utils.AverageMeter("loss", "accuracy")
    valid_meter = utils.AverageMeter("loss", "accuracy")

    # Train
    with Progress() as p:
        logger.info("Start training")

        for i in p.trange(1, config.train.num_epochs + 1, description="Epoch"):
            train_meter.reset()

            for j, (images, labels) in enumerate(p.track(train_loader, description="Train", remove=True)):
                key, model_key = random.split(key)
                rngs = jax_utils.replicate(dict(dropout=model_key))
                state, train_metric = train_step(state, rngs, images=images, labels=labels)
                train_meter.update(train_metric, n=len(images))

            if use_batch_stats:
                state = sync_batch_stats(state)

            if i % config.train.print_every == 0:
                logger.info(
                    f"Epoch {i:3d} / {config.train.num_epochs} | "
                    f"Train Loss: {train_meter.loss:7.4f}  Acc: {train_meter.accuracy * 100:6.2f}"
                )

            if i % config.train.valid_every == 0:
                valid_meter.reset()

                for j, (images, labels) in enumerate(p.track(valid_loader, description="Valid", remove=True)):
                    key, model_key = random.split(key)
                    rngs = jax_utils.replicate(dict(dropout=model_key))
                    valid_metric = valid_step(state, rngs, images=images, labels=labels)
                    valid_meter.update(valid_metric, n=len(images))

                if i % config.train.print_every == 0:
                    logger.info(
                        f"                | "
                        f"Valid Loss: {valid_meter.loss:7.4f}  Acc: {valid_meter.accuracy * 100:6.2f}"
                    )

            if i % config.train.save_every == 0 and jax.process_index() == 0:
                checkpoints.save_checkpoint(
                    output_dir, jax_utils.unreplicate(state),
                    step=i, prefix="checkpoint_", keep=1,
                )

    logger.info("Finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=False, conflict_handler="resolve")
    parser.add_argument("-f", "--config-file", type=str, required=True)
    args, rest_args = parser.parse_known_args()

    config: ConfigDict = load_config(args.config_file).lock()
    add_config_arguments(parser, config, aliases={
        "train.seed":              ["-s",   "--seed"],
        "train.num_epochs":        ["-e",   "--epochs"],
        "dataset.name":            ["-d",   "--dataset"],
        "dataset.batch_size":      ["-bs",  "--batch-size"],
        "model.name":              ["-m",   "--model"],
        "optimizer.name":          ["-opt", "--optimizer"],
        "optimizer.learning_rate": ["-lr",  "--learning-rate"],
    })
    parser.add_argument("-f", "--config-file", default=argparse.SUPPRESS)
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS)
    args = parser.parse_args(rest_args)

    config.update(vars(args))

    # Logger
    log_name = utils.get_experiment_name()
    output_dir = os.path.join("outs", "_", log_name)
    latest_link = os.path.join("outs", "_", "_latest")

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(log_name, latest_link)
    save_config(config, os.path.join(output_dir, "config.yaml"))

    logger = utils.setup_logger(__name__, output_dir, suppress=[jax, flax])
    logger.debug("python " + " ".join(sys.argv))

    args_str = "Configs:"
    for k, v in config.items(flatten=True):
        args_str += f"\n    {k:<25}: {v}"
    logger.info(args_str)

    logger.info(f"Output directory: \"{output_dir}\"")

    try:
        main(config, output_dir)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception(e)
