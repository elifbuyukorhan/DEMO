import os
from typing import Dict

import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm
from models.yolo import Model
import yaml
from utils.datasets import create_dataloader
import torch.utils.data
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr

import ray.train
from ray.train import *
#from ray.train import ScalingConfig
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer
import ray

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

# def get_dataloaders(batch_size):
#     # Transform to normalize the input images
#     transform = transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))])

#     with FileLock(os.path.expanduser("~/data.lock")):
#         # Download training data from open datasets
#         training_data = datasets.MNIST(
#             root='./',
#             train=True,
#             download=True,
#             transform=transform,
#         )

#         # Download test data from open datasets
#         test_data = datasets.MNIST(
#             root='./',
#             train=False,
#             download=True,
#             transform=transform,
#         )


#     # Create data loaders
#     train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(test_data, batch_size=batch_size)

#     return train_dataloader, test_dataloader


def get_dataloaders(batch_size):
    # Caltech101 için dönüşüm
    transform = transforms.Compose([
        transforms.Resize(224),  # Resmi 224x224 boyutuna dönüştürme
        transforms.CenterCrop(224),  # Merkezden kırpmayla 224x224'lük bir alanı al
        transforms.ToTensor(),  # Tensöre dönüştürme
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize etme
    ])

    with FileLock(os.path.expanduser("~/data.lock")):
        # Caltech101 veri kümesini indirme
        caltech_data = datasets.Caltech101(
            root='./',
            download=True,
            transform=transform,
        )

    # Veri kümesini train ve test olarak ayırma
    train_size = int(0.8 * len(caltech_data))
    test_size = len(caltech_data) - train_size
    train_data, test_data = torch.utils.data.random_split(caltech_data, [train_size, test_size])

    # Veri yükleyicileri oluşturma
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader

# Model Definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    nc = 1

    # Trainloader
    # train_dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
    #                                         hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
    #                                         world_size=opt.world_size, workers=opt.workers,
    #                                         image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))

    # Get dataloaders inside the worker training function
    train_dataloader, test_dataloader = get_dataloaders(train_path=train_path, test_path=test_path, batch_size=batch_size)

    # [1] Prepare Dataloader for distributed training
    # Shard the datasets among workers and move batches to the correct device
    # =======================================================================
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)
    

    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))
    model = NeuralNetwork()

    # [2] Prepare and wrap your model with DistributedDataParallel
    # Move the model to the correct GPU/CPU device
    # ============================================================
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Model training loop
    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1:
            # Required for the distributed sampler to shuffle properly across epochs.
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                pred = model(X)
                loss = loss_fn(pred, y)

                test_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()

        test_loss /= len(test_dataloader)
        accuracy = num_correct / num_total

        # [3] Report metrics to Ray Train
        # ===============================
        ray.train.report(metrics={"loss": test_loss, "accuracy": accuracy})


def train_fashion_mnist(epochs, num_workers=2, use_gpu=False):
    global_batch_size = 32

    train_config = {
        "lr": 1e-3,
        "epochs": epochs,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    # Configure computation resources
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    # [4] Start distributed training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result = trainer.fit()
    print(f"Training result: {result}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7-tiny.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.tiny.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 32)')
    opt = parser.parse_args()


    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    train_fashion_mnist(opt.epochs, num_workers=1, use_gpu=True)
