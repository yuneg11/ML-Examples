import sys
sys.path.append(".")

import os
import logging

import torch
from torch import optim
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageNet

from nxml.torch.nn import functional as F
from nxcl.experimental import utils
from nxcl.rich import Progress

from models.cnn import CNN
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200


def get_train_loader(dataset: str, batch_size: int, rank: int, distributed: bool):
    if dataset == "cifar10" or dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.202, 0.199, 0.201)),
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
        ])
        train_dataset = SVHN(root="~/datasets", split="train", download=False, transform=train_transform)
        num_workers = 4

    elif dataset == "imagenet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_dataset = ImageNet(root="~/datasets/ILSVRC2012", split="train", transform=train_transform)
        num_workers = 32

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if distributed:
        num_devices = torch.cuda.device_count()
        batch_size = batch_size // num_devices
        sampler = DistributedSampler(
            train_dataset, num_replicas=num_devices,
            rank=rank, shuffle=True, drop_last=True,
        )
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=(not distributed), drop_last=True,
        num_workers=num_workers, sampler=sampler, #pin_memory=True,
    )
    return train_loader


def get_valid_loader(dataset: str, batch_size: int, rank: int, distributed: bool):
    if dataset == "cifar10" or dataset == "cifar100":
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.202, 0.199, 0.201)),
        ])
        CIFAR = CIFAR10 if dataset == "cifar10" else CIFAR100
        valid_dataset = CIFAR(root="~/datasets", train=False, download=False, transform=valid_transform)
        num_workers = 4

    elif dataset == "svhn":
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.196, 0.198, 0.199)),
        ])
        valid_dataset = SVHN(root="~/datasets", split="test", download=False, transform=valid_transform)
        num_workers = 4

    elif dataset == "imagenet":
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        valid_dataset = ImageNet(root="~/datasets/ILSVRC2012", split="val", transform=valid_transform)
        num_workers = 32

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if distributed:
        num_devices = torch.cuda.device_count()
        batch_size = batch_size // num_devices
        sampler = DistributedSampler(
            valid_dataset, num_replicas=num_devices,
            rank=rank, shuffle=False, #drop_last=True,
        )
    else:
        sampler = None

    valid_loader = DataLoader(
        valid_dataset, batch_size, shuffle=False, #drop_last=True,
        num_workers=num_workers, sampler=sampler, #pin_memory=True,
    )
    return valid_loader


def get_train_step(model, device, optimizer):
    def train_step(*, images, labels):
        model.train()

        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(input=logits, target=labels)
        accuracy = F.accuracy(input=logits, target=labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return dict(loss=loss, accuracy=accuracy)
    return train_step


def get_valid_step(model, device):
    @torch.no_grad()
    def valid_step(*, images, labels):
        model.eval()

        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(input=logits, target=labels)
        accuracy = F.accuracy(input=logits, target=labels)

        return dict(loss=loss, accuracy=accuracy)
    return valid_step


def main(rank, args, output_dir, distributed):
    if distributed:
        dist.init_process_group(
            backend="NCCL",
            init_method=args.dist_url,
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        dist.barrier()
        torch.cuda.set_device(rank)

    if torch.cuda.is_available():
        device = torch.device(type="cuda", index=rank)
    else:
        device = torch.device(type="cpu")

    if rank == 0:
        if distributed:
            logger = utils.setup_logger(__name__, output_dir, suppress=[torch])
        else:
            logger = logging.getLogger(__name__)


    if args.dataset == "cifar10":
        image_shape = (3, 32, 32)
        num_classes = 10
    elif args.dataset == "cifar100":
        image_shape = (3, 32, 32)
        num_classes = 100
    elif args.dataset == "svhn":
        image_shape = (3, 32, 32)
        num_classes = 10
    elif args.dataset == "imagenet":
        image_shape = (3, 224, 224)
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

    model = models[args.model](num_classes=num_classes, image_shape=image_shape).to(device)
    if distributed:
        model = DDP(model, device_ids=[rank])

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Setup output directory
    if rank == 0:
        utils.link_output_dir(output_dir, subnames=(args.dataset, args.model))

    # Setup steps
    train_step = get_train_step(model, device, optimizer)
    valid_step = get_valid_step(model, device)

    train_loader = get_train_loader(args.dataset, args.batch_size, rank, distributed=distributed)
    valid_loader = get_valid_loader(args.dataset, args.batch_size, rank, distributed=distributed)

    train_meter = utils.AverageMeter("loss", "accuracy")
    valid_meter = utils.AverageMeter("loss", "accuracy")

    # Train
    with Progress(disable=(rank!=0)) as p:
        for i in p.trange(1, args.num_epochs + 1, description="Epoch"):

            train_meter.reset()
            for j, (images, labels) in enumerate(p.track(train_loader, description="Train", remove=True)):
                train_metric = train_step(images=images, labels=labels)
                train_meter.update(train_metric, n=len(images))

            valid_meter.reset()
            for j, (images, labels) in enumerate(p.track(valid_loader, description="Valid", remove=True)):
                valid_metric = valid_step(images=images, labels=labels)
                valid_meter.update(valid_metric, n=len(images))

            if rank == 0:
                logger.info(
                    f"Epoch {i:3d} / {args.num_epochs} | "
                    f"Train Loss: {train_meter.loss:7.4f}  Acc: {train_meter.accuracy * 100:6.2f} | "
                    f"Valid Loss: {valid_meter.loss:7.4f}  Acc: {valid_meter.accuracy * 100:6.2f}"
                )

            if i % args.save_every == 0 and rank == 0:
                state = {
                    "epoch": i,
                    "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(), # {k: v.cpu() for k, v in optimizer.state_dict().items()},
                }
                torch.save(state, os.path.join(output_dir, f"checkpoint_{i}.pt"))

    if distributed:
        dist.destroy_process_group()

    if rank == 0:
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
    parser.add_argument("-wd",  "--weight-decay",  type=float, default=0.0)
    parser.add_argument("-s",   "--seed",          type=int,   default=0)
    parser.add_argument("--valid-every",           type=int,   default=100)
    parser.add_argument("--save-every",            type=int,   default=10)
    parser.add_argument("-url", "--dist-url",      type=str,   default="tcp://localhost:12345")
    args = parser.parse_args()

    # Logger
    log_name = utils.get_experiment_name()

    output_dir = os.path.join("outs", "_", log_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = utils.setup_logger(__name__, output_dir, suppress=[torch])
    logger.debug("python " + " ".join(sys.argv))

    args_str = "\n  Arguments:"
    for k, v in vars(args).items():
        args_str += f"\n    {k:<15}: {v}"
    logger.info(args_str + "\n")

    try:
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            mp.spawn(
                main,
                nprocs=num_devices,
                args=(args, output_dir, True),
                start_method="spawn",
            )
        else:
            main(0, args, output_dir, distributed=False)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception(e)
