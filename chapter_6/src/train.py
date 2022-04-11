from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from vit_pytorch.efficient import ViT

# Import SMDataParallel PyTorch Modules
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
import smdistributed.dataparallel.torch.distributed as dist

# step 1: initialize process group
dist.init_process_group()

seed = 42

class CustomImageLoader(datasets.ImageFolder):
    def __getitem__(self, idx):
        img_path = self.imgs[idx][0]
        image, label = super(CustomImageLoader, self).__getitem__(idx)
        return image, label
    
    def __len__(self):
        return len(self.imgs)
    
# function to set the seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

def get_train_data_loader(train_dir, batch_size, world_size, rank):
    print("--------- Training Initializing Dataloader ----------------")

    train_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(0.2),
                                    transforms.RandomGrayscale(),
                                    transforms.RandomVerticalFlip(0.2),
                                    transforms.RandomInvert(0.1),
                                    transforms.ToTensor()])
    
    
    train_dataset = CustomImageLoader(root=train_dir, transform=train_transform)

    print('---------------- Train Dataset Info:-----------------\n', train_dataset)
    
    # step 2: training dataloader should have DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=False, 
                                         drop_last=True,
                                         pin_memory=True,
                                         sampler=train_sampler,
                                        )

    
    print('Train dataset and data loader length: ', len(train_dataset), len(train_dataloader))
    return train_dataloader

def get_validation_data_loader(val_dir, batch_size, rank):
    print("--------- Validation Initializing Dataloader ----------------")
    
    val_transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    
    validation_dataset = CustomImageLoader(root=val_dir, transform=val_transform)
    print('---------------- Validation Dataset Info:-----------------\n', validation_dataset)
    
    val_dataloader = torch.utils.data.DataLoader(validation_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=True, 
                                             drop_last=True,
                                            )
    print('Validation dataset and data loader length: ',len(validation_dataset), len(val_dataloader))
    
    return val_dataloader

    

def train(args, criterion, optimizer, train_dataloader, model, epoch, rank, world_size):
    model.train()
    epoch_accuracy = 0
    epoch_loss = 0
    for batch_idx, (data, label) in enumerate(train_dataloader):
        data = data.to(args.device)
        label = label.to(args.device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and rank == 0:
            print('Batch idx: ', batch_idx)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data) * world_size,
                    len(train_dataloader.dataset),
                    100.0 * batch_idx / len(train_dataloader),
                    loss.item(),
                )
            )
        if args.verbose:
            print("Batch", batch_idx, "from rank", rank)

#         acc = (output.argmax(dim=1) == label).float().mean()
#         epoch_accuracy += acc / len(train_dataloader)
#         epoch_loss += loss / len(train_dataloader)
#     if rank == 0:
#         print(
#             f"Epoch : {epoch} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n"
#         )

def test(model, criterion, device, val_dataloader, epoch):
    model.eval()
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    with torch.no_grad():
        for data, label in val_dataloader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_dataloader)
            epoch_val_loss += val_loss / len(val_dataloader)
        print( f"Epoch : {epoch} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

def save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)
    print(f"Saving model: {path} \n")
    
def main():
    parser = argparse.ArgumentParser()

    # parsing the hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--subset', type=int, default=250)
    parser.add_argument('--num_workers', type=int, default=4) # for multi-process loading 
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_batch_size', type=int, default=64)

    # PyTorch container environment variables for data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--save-model", action="store_true", default=True, help="For Saving the current Model")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="For displaying smdistributed.dataparallel-specific logs",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args, _ = parser.parse_known_args()
    
    print('training directory path: ', args.train)
    
    #step 3: based on the instance used, get the worldsize = total number of GPUs available, rank and local_rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = dist.get_local_rank()
    args.device = device = torch.device("cuda")
    
    # step 4: define batch size based on world size, divide the gobal batch size to distribute data between GPUs
    args.batch_size //= world_size // 8
    args.batch_size = max(args.batch_size, 1)
    train_dir = args.train
    val_dir = args.val
#     print('------------ Batch Size -----------', args.batch_size)
    if args.verbose:
        print(
            "Hello from rank",
            rank,
            "of local_rank",
            local_rank,
            "in world size of",
            world_size,
        )
    # get training and validation data loaders
    train_dataloader = get_train_data_loader(train_dir, args.batch_size, world_size, rank)
    if rank == 0:
        validation_dataloader = get_validation_data_loader(val_dir, args.test_batch_size, rank)
    
    efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64)
    
    # download pre-trained ViT model 
    model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,).to(args.device)
    
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, criterion, optimizer, train_dataloader, model, epoch, rank, world_size)
        if rank == 0:
            test(model, criterion, args.device, validation_dataloader, epoch)
        scheduler.step()

    if rank == 0:
        model_save = model.module if hasattr(model, "module") else model
        save_model(model_save, args.model_dir)
#         torch.save(model.state_dict(), os.path.join(args.model-dir, "vit.pt"))
    
if __name__ =='__main__':
    main()
