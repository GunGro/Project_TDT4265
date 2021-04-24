#! /bin/env python
import torch
from DatasetLoader import DatasetLoader
from pathlib import Path
from utils import load_model
from torch.utils.data import DataLoader
from train import calculate_dice, batch_to_img, predb_to_mask
import matplotlib.pyplot as plt


def load_TEE(bs):

    base_path = Path("../datasets/TEE_GT/")
    data = DatasetLoader(base_path/'train_gray',
                        base_path/'train_gt')
    data.TEE = True
    test_data = DataLoader(data,batch_size=bs)

    return test_data


if __name__ == '__main__':
    visual_debug = True

    bs = 8

    model = load_model(Path('../files/ModelCache/LeadingModel.pt'))

    test_data = load_TEE(bs)


    running_loss = 0
    running_acc = [0]*3
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in iter(test_data):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            model.train(False)
            outputs = model(x)
            running_loss += loss_fn(outputs, y).item() * y.shape[0]
            DSC = calculate_dice(outputs, y)
            for i in range(len(running_acc)):
                running_acc[i] += DSC[i] * y.shape[0]

        for i in range(len(running_acc)):
            running_acc[i] /= len(test_data.dataset)

    print(f"Test loss: {running_loss / len(test_data.dataset):.4f}")
    for i, dice_score in enumerate(DSC):
        print(f"Dice score of class {i}: {dice_score : .4f}")

    xb, yb = next(iter(test_data))
    with torch.no_grad():
        model.cpu()
        predb = model(xb)
    if visual_debug:
        fig, ax = plt.subplots(bs, 3, figsize=(15, bs * 5))
        for i in range(bs):
            ax[i, 0].imshow(batch_to_img(xb, i))
            ax[i, 1].imshow(yb[i])
            ax[i, 2].imshow(predb_to_mask(predb, i))

        plt.show()



