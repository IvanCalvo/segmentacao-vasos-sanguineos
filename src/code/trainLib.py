import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from IPython import display
from dataset import get_dataset

def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed) 
    np.random.seed(seed)

def show_log(logger):
    epochs, losses_train, losses_valid, accs, dice_scores = zip(*logger)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9,3))
    ax1.plot(epochs, losses_train, '-o', label='Train loss')
    ax1.plot(epochs, losses_valid, '-o', label='Valid loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim((0,1.))
    ax1.legend()
    ax2.plot(epochs, accs, '-o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim((0,1.))
    ax3.plot(epochs, dice_scores, '-o')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('clDice Score')
    ax3.set_ylim((0,1.))
    fig.tight_layout()

    display.clear_output(wait=True) 
    plt.show()

def train_step(model, dl_train, optim, loss_func, scheduler, device):
    model.train()
    loss_log = 0.
    for imgs, targets in dl_train:
        imgs = imgs.to(device)
        targets = targets.to(device)
        model.zero_grad()
        scores = model(imgs)#['out']
        loss = loss_func(scores, targets)
        loss.backward()
        optim.step()
        
        loss_log += loss.detach()*imgs.shape[0]

    scheduler.step()

    loss_log /= len(dl_train.dataset)

    return loss_log.item()

@torch.no_grad()
def accuracy(scores, targets):
    return (scores.argmax(dim=1)==targets).float().mean()

@torch.no_grad()
def valid_step(model, dl_valid, loss_func, perf_func, dice_func, device):

    model.eval()

    loss_log = 0.
    perf_log = 0.
    dice_log = 0.
    for imgs, targets in dl_valid:
        imgs = imgs.to(device)
        targets = targets.to(device)
        scores = model(imgs)#['out']
        loss = loss_func(scores, targets)
        perf = perf_func(scores, targets)
        dice = dice_func(scores, targets)

        loss_log += loss*imgs.shape[0]
        perf_log += perf*imgs.shape[0]
        dice_log += dice*imgs.shape[0]

    loss_log /= len(dl_valid.dataset)
    perf_log /= len(dl_valid.dataset)
    dice_log /= len(dl_valid.dataset)

    return loss_log.item(), perf_log.item(), dice_log.item()

def train(model, bs, num_epochs, lr, weight_decay=0., resize_size=256, seed=0, 
          num_workers=5):
    

    seed_all(seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    ds_train, ds_valid, class_weights = get_dataset('../data/VessMAP', resize_size=resize_size)

    model.to(device)

    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, 
                          num_workers=num_workers, persistent_workers=num_workers>0)
    dl_valid = DataLoader(ds_valid, batch_size=bs, shuffle=False,
                          num_workers=num_workers, persistent_workers=num_workers>0)

    loss_func = nn.CrossEntropyLoss(torch.tensor(class_weights, device=device))
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                            momentum=0.9)
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
    logger = []
    best_loss = torch.inf
    for epoch in range(0, num_epochs):
        loss_train = train_step(model, dl_train, optim, loss_func, sched, device)
        loss_valid, perf = valid_step(model, dl_valid, loss_func, accuracy, device)
        logger.append((epoch, loss_train, loss_valid, perf))

        show_log(logger)


        checkpoint = {
            'params':{'bs':bs,'lr':lr,'weight_decay':weight_decay},
            'model':model.state_dict(),
            'optim':optim.state_dict(),
            'sched':sched.state_dict(),
            'logger':logger
        }


        torch.save(checkpoint, '../data/checkpoints/M06/checkpoint.pt') 


        if loss_valid<best_loss:
            torch.save(checkpoint, '../data/checkpoints/M06/best_model.pt')
            best_loss = loss_valid             

    model.to('cpu')

    return ds_train, ds_valid, logger
