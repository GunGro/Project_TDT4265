#!/usr/bin/env python 

import numpy as np
import sys
import matplotlib as mp
import matplotlib.pyplot as plt
import time

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

from DatasetLoader import DatasetLoader
from Unet2D import Unet2D


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()

    if torch.cuda.is_available():
        model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print("-"*20)
        print()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                step += 1
                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*y.shape[0]#dataloader.batch_size
                running_loss += loss*y.shape[0]#dataloader.batch_size 

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_acc / len(dataloader.dataset)
        
            print('Epoch: {:.0f} {} Loss: {:.4f} Acc: {}'.format(epoch, phase, epoch_loss, epoch_acc) + " "*20)

            train_loss.append(epoch_loss.item()) if phase=='train' else valid_loss.append(epoch_loss.item())
        if epoch < epochs -1:
            print("\033[F"*5, end = "")

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    

def acc_metric(predb, yb):
    if torch.cuda.is_available():
        return (predb.argmax(dim=1) == yb.cuda()).float().mean()
    else:
        return (predb.argmax(dim=1) == yb).float().mean()

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

# antar da at vi har y / y_pred som ser slik ut: (batch/index of image, height, width, class_map)
def calculate_dice(predb, yb):
    num_classes = predb.shape[1]
    predb = predb.argmax(dim = 1)
    DSC_vec = []
    for i in range(1, num_classes):
        intersection =((yb == i)*(predb == i)).sum()
        smooth = 1e-8
        DSC = (2 * intersection + smooth) / ((yb == i).sum()+ (predb == i).sum() + smooth)
        DSC_vec.append(DSC.item())
    return DSC_vec

def multiclass_dice(y, y_pred, num_classes):
    dice=0
    for index in range(num_classes):
        dice += calculate_dice(y[:,:,:,index], y_pred[:,:,:,index])
    return dice/num_classes


def main ():
    #enable if you want to see some plotting
    visual_debug = True

    #batch size
<<<<<<< HEAD
    bs = 15
=======
    bs = 12
>>>>>>> refs/remotes/origin/main

    #epochs
    epochs_val = 100
    
    # set gca to "AKtgg"
    mp.use("TkAgg")
    
    #learning rate
    learn_rate = 0.01

    #load the training data
    base_path = Path("../datasets/CAMUS_full/")
    data = DatasetLoader(base_path/'train_gray', 
                        base_path/'train_gt')
    print(len(data))
    print(type(len(data))) 
    assert len(data) % 3 == 0, f"dude, skjerp deg"

    num_train = int(0.6 * len(data))
    num_val   = int(0.3 * len(data))
    num_test = len(data) - num_train - num_val

    #split the training dataset and initialize the data loaders
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(data, (num_train, num_val, num_test))
    train_data = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=bs, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=bs)

    if visual_debug:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data.open_as_array(150))
        ax[1].imshow(data.open_mask(150))
        plt.show()

    xb, yb = next(iter(train_data))
    print (xb.shape, yb.shape)

    # build the Unet2D with one channel as input and 2 channels as output
    unet = Unet2D(1,4)
    print(unet)
    print("neste eksempel")
    for i, child in enumerate(unet.children()):
        print(i, child)

    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    #opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)
    opt = torch.optim.Adam(unet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    #do some training
    train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, acc_metric, epochs=epochs_val)

    #plot training and validation losses
    if visual_debug:
        plt.figure(figsize=(10,8))
        plt.plot(train_loss, label='Train loss')
        plt.plot(valid_loss, label='Valid loss')
        plt.legend()
        plt.show()

    #predict on the test dataset 

    running_loss = 0.0
    running_acc  = [0]*3

    with torch.no_grad():
        for x, y in iter(test_data):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            unet.train(False)
            outputs = unet(x)
            running_loss += loss_fn(outputs, y).item()*y.shape[0]
            DSC = calculate_dice(outputs, y)
            for i in range(len(running_acc)):
                running_acc[i]  += DSC[i]*y.shape[0]
                
        for i in range(len(running_acc)):
            running_acc[i] /= len(test_data.dataset)

    print(f"Test loss: { running_loss/len(test_data.dataset):.4f}")
    for i, dice_score in enumerate(DSC):
        print(f"Dice score of class {i}: {dice_score : .4f}")
        
    #show the predicted segmentations

    xb, yb = next(iter(train_data))
    with torch.no_grad():
        if torch.cuda.is_available():
            predb = unet(xb.cuda())
        else:
            predb = unet(xb)
    
    if visual_debug:
        fig, ax = plt.subplots(bs,3, figsize=(15,bs*5))
        for i in range(bs):
            ax[i,0].imshow(batch_to_img(xb,i))
            ax[i,1].imshow(yb[i])
            ax[i,2].imshow(predb_to_mask(predb, i))

        plt.show()
    calculate_dice(predb.cpu(), yb.cpu())

if __name__ == "__main__":
    main()
