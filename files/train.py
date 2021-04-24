#!/usr/bin/env python 

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import time

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn


from DatasetLoader import DatasetLoader
from Unet2D import Unet2D
from utils import load_model, save_model


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1, do_mixup = False):
    start = time.time()


    if torch.cuda.is_available():
        model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0
    checkpoint = 0

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
                if do_mixup and y.shape[0] > 1 and phase == 'train':
                    P = np.random.choice(y.shape[0],size=y.shape[0],replace = False)
                    t = np.random.beta(0.3,0.3,1); t = max(t,1-t).item()
                    x_shuffle = x[P]
                    y_shuffle = y[P]
                    x = t*x + (1-t)*x_shuffle
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y_shuffle = y_shuffle.cuda()
                        y = y.cuda()
                else:
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()

                step += 1
                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    if do_mixup:
                        loss = t*loss_fn(outputs, y) + (1-t)*loss_fn(outputs, y_shuffle)
                    else:
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


            if phase == 'valid' and len(valid_loss) > 0 and epoch_loss.item() < min(valid_loss) and epoch > 10:
                save_model(model, Path('../files/ModelCache/LeadingModel.pt'))
                checkpoint = epoch



            train_loss.append(epoch_loss.item()) if phase=='train' else valid_loss.append(epoch_loss.item())

            # do early stop?
            if phase == 'valid' and (epoch - checkpoint) == 25:
                print('Stopping early')

                time_elapsed = time.time() - start
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

                train_loss.append(epoch_loss.item()) if phase == 'train' else valid_loss.append(epoch_loss.item())
                return train_loss, valid_loss



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


def main (do_augment, do_mixup, do_blur):
    #enable if you want to see some plotting
    visual_debug = True
    stupid_visual_debug = False

    #batch size
    bs = 12

    #epochs
    epochs_val = 4000
    
    # set gca to "AKtgg"
    mp.use("TkAgg")
    
    #learning rate
    learn_rate = 0.01

    #load the training data
    base_path = Path("../datasets/CAMUS_full/Train/")
    data = DatasetLoader(base_path/'train_gray', 
                        base_path/'train_gt')


    num_train = int(0.6 * len(data))
    num_val   = len(data) - num_train



    #split the training dataset and initialize the data loaders
    train_dataset, valid_dataset = torch.utils.data.random_split(data, (num_train, num_val))
    train_data = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=bs, shuffle=True)

    if stupid_visual_debug:
        xb, yb = next(iter(train_data))
        data.do_augment = do_augment
        data.do_blur = do_blur
        for x,y in zip (xb, yb):
            fig, ax = plt.subplots(1,2)
            xb, yb = next(iter(train_data))
            ax[0].imshow(xb[0,0].numpy())
            ax[1].imshow(yb[0].numpy())
            plt.show()
        data.do_augment = False
        data.do_blur = False

    xb, yb = next(iter(train_data))

    # build the Unet2D with one channel as input and 2 channels as output
    unet = Unet2D(1,4)

    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)
    # opt = torch.optim.Adam(unet.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    #consider trying with sgd also:
    opt = torch.optim.SGD(unet.parameters(), lr=learn_rate, momentum = 0.9)


    #do some training
    data.do_blur = do_blur
    data.do_augment = do_augment
    train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, acc_metric, epochs=epochs_val, do_mixup=do_mixup)
    data.do_blur = False
    data.do_augment = False

    #plot training and validation losses
    if visual_debug:
        plt.figure(figsize=(10,8))
        plt.plot(train_loss, label='Train loss')
        plt.plot(valid_loss, label='Valid loss')
        plt.legend()
        plt.show()

    #predict on the test dataset 
    base_path = Path("../datasets/CAMUS_full/Test/")
    data = DatasetLoader(base_path/'train_gray',
                        base_path/'train_gt')

    test_data = DataLoader(data,batch_size=bs)


    running_loss = 0.0
    running_acc  = [0]*3

    test_model = load_model(Path('../files/ModelCache/LeadingModel.pt'))


    with torch.no_grad():
        for x, y in iter(test_data):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            test_model.train(False)
            outputs = test_model(x)
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

    xb, yb = next(iter(test_data))
    with torch.no_grad():
        if torch.cuda.is_available():
            predb = test_model(xb.cuda())
        else:
            predb = test_model(xb)

    if visual_debug:
        fig, ax = plt.subplots(bs, 3, figsize=(15, bs * 5))
        for i in range(bs):
            ax[i, 0].imshow(batch_to_img(xb, i))
            ax[i, 1].imshow(yb[i])
            ax[i, 2].imshow(predb_to_mask(predb, i))

        plt.show()

if __name__ == "__main__":
    main(do_augment = True, do_mixup = False, do_blur = False)
