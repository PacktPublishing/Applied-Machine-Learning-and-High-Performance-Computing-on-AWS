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
from torchnet.dataset import SplitDataset

# smdistributed: import library
import smdistributed.modelparallel
import smdistributed.modelparallel.torch as smp

# smdistributed: initialize the backend
smp.init()

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

def get_train_data_loader(train_dir, batch_size):
    print("--------- Training Initializing Dataloader ----------------")
    train_transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(0.2),
                                        transforms.RandomGrayscale(),
                                        transforms.RandomVerticalFlip(0.2),
                                        transforms.RandomInvert(0.1),
                                        transforms.ToTensor()])
    if smp.pp_rank() == 0:

        train_dataset = CustomImageLoader(root=train_dir, transform=train_transform)

        # smdistributed: Shard the dataset based on data-parallel ranks
        train_sampler = torch.utils.data.DistributedSampler(
                train_dataset,
                shuffle=False,
                seed=seed,
                rank=smp.dp_rank(), # The rank of the process within its data-parallelism group.
                num_replicas=smp.dp_size(), # The size of the data-parallelism group.
                drop_last=True,
            )

        print('---------------- Train Dataset Info:-----------------\n', train_dataset)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False, 
                                             drop_last=True,
                                             pin_memory=True,
                                             num_workers=0,
                                             sampler=train_sampler,
                                            )
        smp.broadcast(len(train_dataloader), smp.PP_GROUP)
    else: 
        data_len = smp.recv_from(0, smp.RankType.PP_RANK)
        train_dataset = CustomImageLoader(root=train_dir, transform=train_transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    print(f'DP Rank {smp.dp_rank()} Training dataset and data loader length: ',len(train_dataset), len(train_dataloader))
    return train_dataloader

def get_validation_data_loader(val_dir, batch_size):
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
                                             num_workers=0,
                                            )
    print('Validation dataset and data loader length: ',len(validation_dataset), len(val_dataloader))
    
    return val_dataloader

# smdistributed: Define smp.step. Return any tensors needed outside.
@smp.step
def train_step(model, data, label):
    output = model(data)
    loss = F.nll_loss(F.log_softmax(output), label, reduction="mean")
    # replace loss.backward() with model.backward in the train_step function.
    model.backward(loss)
    return output, loss
    

def train(args, optimizer, train_dataloader, model, epoch):
    model.train()
    epoch_accuracy = 0
    epoch_loss = 0
    for batch_idx, (data, label) in enumerate(train_dataloader):
        # smdistributed: Move input tensors to the GPU ID used by the current process,
        # based on the set_device call.
        data = data.to(args.device)
        label = label.to(args.device)
        
        optimizer.zero_grad()
        _, loss_mb = train_step(model, data, label)
        # smdistributed: Average the loss across microbatches.
        loss = loss_mb.reduce_mean()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and smp.rank() == 0:
            print(f"(Epoch: {epoch} Batch: {batch_idx} train_loss: {loss.item()})")
        if args.verbose:
            print("Batch", batch_idx, "from rank", smp.rank())

# smdistributed: Define smp.step. Return any tensors needed outside.
@smp.step
def test_step(model, data, label):
    val_output = model(data)
    val_loss = F.nll_loss(F.log_softmax(val_output), label, reduction="mean")
    return val_loss

def test(model, device, val_dataloader, epoch):
    model.eval()
    val_loss = 0.0
    n_batches = 0
    num_batches = len(val_dataloader)
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_dataloader):
            if batch_idx >= num_batches:
                print('------------ Breaking out of the loop -------')
                break
            data = data.to(device)
            label = label.to(device)
            val_loss += test_step(model, data, label).reduce_mean()
            n_batches += 1

    if n_batches > 0:
        val_loss /= n_batches
        val_loss = val_loss.item()
    if n_batches == 0:
        val_loss = -1.0
    print( f"Epoch : {epoch} - val_loss : {val_loss:.4f}\n")

def save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pth')
    # # smdistributed: smp.save() - saves an object. This operation is similar to torch.save().
    smp.save(model.state_dict(), path)
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
    parser.add_argument('--test_batch_size', type=int, default=8)

    # PyTorch container environment variables for data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument("--save-model", action="store_true", default=True, help="For Saving the current Model")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
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
 
    train_dir = args.train
    val_dir = args.val
    # get training and validation data loaders
    train_dataloader = get_train_data_loader(train_dir, args.batch_size)
    if smp.rank() == 0:
        print('-------------- Gettting Validation Data Loader -------------------')
        validation_dataloader = get_validation_data_loader(val_dir, args.test_batch_size)


    efficient_transformer = Linformer(
        dim=1536,
        seq_len=49+1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=12,
        k=64)

    # download pre-trained ViT model 
    model = ViT(
        dim=1536,
        image_size=224,
        patch_size=32,
        num_classes=2,
        transformer=efficient_transformer,
        channels=3,)
    
    # print number of parameters - should be 344804354 approx 344M 
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if smp.rank() == 0:
        print('number of params:', n_parameters)
    
    # smdistributed: Use the DistributedModel container to provide the model
    # to be partitioned across different ranks. For the rest of the script,
    # the returned DistributedModel object should be used in place of
    # the model provided for DistributedModel class instantiation.
    
    model = smp.DistributedModel(model)
    # optimizer
    optimizer = smp.DistributedOptimizer(optim.Adam(model.parameters(), lr=args.lr))
    
     # smdistributed: Set the device to the GPU ID used by the current process.
    # Input tensors should be transferred to this device.
    torch.cuda.set_device(smp.local_rank())
    args.device = device = torch.device("cuda")
    
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, optimizer, train_dataloader, model, epoch)
        if smp.rank() == 0:
            print(f'-------------- Start Model Evaluation for the Epoch {epoch} -------------------')
            test(model, args.device, validation_dataloader, epoch)
        scheduler.step()

    if smp.rank() == 0:
        model_save = model.module if hasattr(model, "module") else model
        save_model(model_save, args.model_dir)
    
if __name__ =='__main__':
    main()
