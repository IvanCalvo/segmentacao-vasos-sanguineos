import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_dataset
import trainLib as train_class 
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np

# As próximas linhas devem ser descomentadas caso haja a necessidade de utilizar o tensorboard
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter(comment='-torchseg - Unet - densenet121')

@torch.no_grad()
def iou(scores, targets):
    pred = scores.argmax(dim=1).reshape(-1)
    targets = targets.reshape(-1)
    
    pred = pred[targets!=2]
    targets = targets[targets!=2]

    tp = ((pred==1) & (targets==1)).sum()
    fp = ((pred==1) & (targets==0)).sum()
    fn = ((pred==0) & (targets==1)).sum()
    iou = tp/(tp+fp+fn)

    return iou

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def clDiceFunc(scores, targets):
    pred_list = []
    target_list = []
    cl_dice_list = []
    
    scores = scores.to('cpu')
    targets = targets.to('cpu')

    for score in scores:
        pred = score.argmax(dim=0)
        pred = pred.numpy()
        pred_list.append(pred)

    for target in targets:
        trg = target.numpy()
        target_list.append(trg)

    for pred, target in zip(pred_list, target_list):
        clDice_score = clDice(pred, target)
        cl_dice_list.append(clDice_score)
    
    return (sum(cl_dice_list)/len(cl_dice_list))

def train(model, bs_train, bs_valid, num_epochs, lr, weight_decay=0., resize_size=256, seed=0, 
          num_workers=5, model_name='Model'):
    
    
    train_class.seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ds_train, ds_valid, class_weights = get_dataset(
        '../data/VessMAP', split=0.88, resize_size=resize_size
        )
    #ds_train.indices = ds_train.indices[:5*256]
    model.to(device)

    dl_train = DataLoader(ds_train, batch_size=bs_train, shuffle=True, 
                          num_workers=num_workers, persistent_workers=num_workers>0)
    
    dl_valid = DataLoader(ds_valid, batch_size=bs_valid, shuffle=False,
                          num_workers=num_workers, persistent_workers=num_workers>0)

    loss_func = nn.CrossEntropyLoss(torch.tensor(class_weights, device=device), ignore_index=2)
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                            momentum=0.9) 
    
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
    logger = []
    best_loss = torch.inf
    
    for epoch in range(0, num_epochs):
        loss_train = train_class.train_step(model, dl_train, optim, loss_func, sched, device)
        loss_valid, perf, dice_score = train_class.valid_step(model, dl_valid, loss_func, iou, clDiceFunc, device)
        logger.append((epoch, loss_train, loss_valid, perf, dice_score))
        '''writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('Loss/test', loss_valid, epoch)
        writer.add_scalar('iou', perf, epoch)
        writer.add_scalar('clDice', dice_score, epoch)'''

        train_class.show_log(logger)

        checkpoint = {
            'params':{'bs_train':bs_train,'bs_valid':bs_valid,'lr':lr,
                      'weight_decay':weight_decay},
            'model':model.state_dict(),
            'optim':optim.state_dict(),
            'sched':sched.state_dict(),
            'logger':logger
        }

        torch.save(checkpoint, f'../data/checkpoints/torchseg/checkpoint_{model_name}.pt') 

        if loss_valid<best_loss:
            torch.save(checkpoint, f'../data/checkpoints/torchseg/best_model_{model_name}.pt')
            best_loss = loss_valid       

    # tensorboard
    #writer.flush()

    model.to('cpu')

    return ds_train, ds_valid, logger