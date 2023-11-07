import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import r2_score
from tqdm import tqdm
from Calculating_norms import Load_Normalization_AiT
import gc
# torch.set_float32_matmul_precision('high')
# import warnings
# warnings.filterwarnings("ignore")
# import os
# os.environ["OMP_NUM_THREADS"] = "1"



def standard_x(data, mean, std):
    data = (data - mean.reshape(1, -1, 1, 1, 1)) / std.reshape(1, -1, 1, 1, 1)
    return data


def standard_y(data, mean, std):
    return (data - mean) / std


# define dataset and dataloader
class Mydataset_cv(Dataset):

    def __init__(
        self,
        config,
        mode,
        sample_idx,
        cv,
    ):
        # 读取数据
        path = config['out_path']
        t_scale = config['t_scale']
        wid_s, wid_t = config['wid_s'], config['wid_t']
        labels = config['labels']
        res = config['res']

        data = torch.load('{}data/Training_{}/{}*{}_{}/Feature{}.pt'.format(path, t_scale, wid_s, wid_t, str(res), config['suffix']))
        target = torch.load('{}data/Training_{}/{}*{}_{}/Target{}.pt'.format(path, t_scale, wid_s, wid_t, str(res), config['suffix']))
        # 使用标签全部存在的数据进行训练
        coords = target[:, -2:]

        # 获取想要的污染物作为标签
        target = target[:, :len(labels)]

        # load normalization params
        x_mean, x_std, y_mean, y_std = Load_Normalization_AiT(config)
        # y_mean, y_std = torch.nanmean(target, axis=0), torch.Tensor(np.nanstd(target.numpy(), axis=0))
        data, target = standard_x(data, x_mean, x_std), standard_y(target, y_mean, y_std)
        # 划分训练集、验证集和测试集
        if mode == 'train':
            data, target, coords = data[sample_idx, ::].float(), target[sample_idx, :].float(), coords[sample_idx, :].float()

        if mode == 'valid':
            data, target, coords = data[sample_idx, ::].float(), target[sample_idx, :].float(), coords[sample_idx, :].float()


        self.data = torch.FloatTensor(data)
        # 缺失值转化为0
        target[target != target] = -100
        self.target = torch.FloatTensor(target)
        self.dim = self.data.shape
        print('Finished reading the Dataset ({} samples found, each dim = {}, target dim = {})'.format(len(self.data), self.dim, self.target.shape))

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def __len__(self):
        return len(self.data)


def Mydataloader_cv(
    config,
    n_jobs=0,
    shuffle=True,
    tr_idx=None,
    dv_idx=None,
    cv=0,
):
    dataloader = []
    for m, idx in zip(['train', 'valid'], [tr_idx, dv_idx]):
        dataset = Mydataset_cv(
            config,
            mode=m,
            sample_idx=idx,
            cv=cv,
        )
        dataloader.append(DataLoader(dataset, config['batch_size'], shuffle=shuffle, num_workers=n_jobs, pin_memory=False, drop_last=True))
    return dataloader


# learning rate schedule: warmup
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
      Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

  Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # print(current_step)
        # warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # decadence
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



# 自定义MSE损失函数, 有缺失值的的部分进行mask, loss归置为0
class CustomMSELoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, pred, target):
        mask = (target != -100).float()  # 缺失值为-1,使用mask屏蔽
        loss = torch.sum((pred - target)**2 * mask)
        num = torch.sum(mask)  # 统计不存在缺失值的元素个数
        return loss / num


# 自定义MSE损失函数, 有缺失值的的部分进行mask, loss归置为0, 可对不同任务自定义损失权重
class CustomMSELoss_weight(nn.Module):
    def __init__(self, lens, device, labels, train_labels, weights=None,):
        super().__init__()
        if weights is None:
            self.weights = torch.ones((lens)).to(device)
        else:
            self.weights = torch.Tensor(weights).to(device)
        self.lens = lens
        self.train_labels_idx = [labels.index(i) for i in train_labels]  # support training in different phases

    def forward(self, pred, target):
        H, W = pred.size()
        mask = (target != -100).float()  # 缺失值为-100,使用mask屏蔽
        loss = torch.sum((pred - target)**2 * mask * self.weights.repeat(H).reshape(H, W), axis=0)
        num = torch.sum(mask, axis=0)  # 统计不存在缺失值的元素个数
        
        return (loss / num)[self.train_labels_idx]          


def autocast_backward(scaler, net, optimizer, X, y, loss_layer):
    optimizer.zero_grad()  # set gradient to zero
    with autocast():
        pred = net(X)  # forward pass (compute output)
        mse_loss = loss_layer(pred, y)  # compute loss
        mse_loss = torch.mean(mse_loss)

    scaler.scale(mse_loss).backward()  # scales the loss, and calls backward to create scaled gradients
    scaler.step(optimizer)  # unscaler gradients and calls or skips optimizer.step()
    scaler.update()  # update the scale for the next iteration
    return mse_loss, pred


# Training model
# @profile
def train(tr_set, dv_set, config, strict=True):

    path = config['out_path']
    res = config['res']
    device = config['device']
    model = config['model'].to(device)
    model_name = config['model_name']
    warmup_steps = config['warmup_steps']
    label = config['labels']
    t_scale = config['t_scale']
    save_path = '{}model/{}_{}/'.format(path, t_scale, res)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    config['model_path'] = save_path + '{}.pth'.format(model_name)
    print('Path of saving model:', config['model_path'])
    """Training: Initialization"""
    # Tensorboard 实例化
    writer = SummaryWriter('{}model/log_{}/{}'.format(path, t_scale, model_name))

    n_epochs = config['n_epochs']
    # 均方误差损失 Mean Squared error loss
    criterion = torch.nn.MSELoss(reduction='mean')
    # criterion = CustomMSELoss_weight(lens=len(label), weights=None, labels=label, train_labels=config['Train_labels'], device=device)
    # criterion = CustomMSELoss_weight(lens=1, weights=None, device=device)
    # criterion = UncertaintyLoss(len(label)).to(device)
    # setup optimizer
    # optimizer = getattr(torch.optim, config['optimizer'])(list(model.parameters()) + list(criterion.parameters()), **config['optim_hparas'])
    if config['Fix_encoder']:
        params = []
        for k, v in model.named_parameters():
            if k.split('.')[0] != 'encoder':
                print(k)
                params.append(v)
            else:
                v.requires_grad = False
        optimizer = getattr(torch.optim, config['optimizer'])(params, **config['optim_hparas'])
    else:
        optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    
    # lr scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, n_epochs)
    min_mse = 10000
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0

    # 断点续训
    if os.path.exists(config['model_path']):
        print('load checkpoint from %s' % config['model_path'])
        if config['PreTraining']:
            checkpoint = torch.load(config['model_path'], map_location='cpu')
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items() if k.split('.')[0] == 'encoder'}, strict=False)
        else:
            checkpoint = torch.load(config['model_path'], map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=strict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch'] + 1
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, n_epochs, last_epoch=epoch)

    # 将模型写入Tensorboard
    # init_data, init_target = next(iter(tr_set))
    # writer.add_graph(model, init_data.to(device))
    # compiled_model = torch.compile(model)
    compiled_model = model

    while epoch < n_epochs:
        model.train()  # set model to training mode
        scaler = GradScaler()
        total_loss = 0
        tr_list = []
        #  use enumerate(tqdm(x)) instead of tqdm(enumerate(x)) to avoid memory leakage issues
        with tqdm(tr_set, total=len(tr_set), leave=False, ncols=120) as loop:
            for index, (x, y) in enumerate(loop):  # iterate through the dataloader
                if (y[:, -1] == -100).sum() != y.size()[0]:
                    x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
                    # PCGrad loss
                    # mse_loss, pred = pcgrad_fn_backward(compiled_model, optimizer, x, y, loss_layer=criterion)
                    # MES mean loss
                    mse_loss, pred = autocast_backward(scaler, compiled_model, optimizer, x, y, loss_layer=criterion)
                    mse_loss_np = mse_loss.detach().cpu().item()
                    pred_np = pred.detach().cpu().numpy()
                    loop.set_description(f'Epoch [{epoch}/{n_epochs}]')  # set tqdm describution
                    loop.set_postfix(loss=mse_loss_np)
                    total_loss += mse_loss_np * len(x)  # accumulate loss
                    tr_list.append(np.concatenate([y.cpu().numpy(), pred_np], axis=1))  # accumulate predicting results
        
        tr_list = np.concatenate(tr_list, axis=0)
        tr_mse = total_loss / tr_list.shape[0]  # compute averaged loss          
        # 获取不含-100的行的索引
        tr_r2 = [r2_score(tr_list[np.argwhere(tr_list[:, i] != -100), i], tr_list[np.argwhere(tr_list[:, i] != -100), i + len(label)]) for i in range(len(label))]  # compute average r2
        loss_record['train'].append(tr_mse)  # add loss to record
        scheduler.step()  # update learning rate

        # after each epoch, test model on the validation set.
        model.eval()
        total_loss = 0
        dv_list = []
        for x, y in dv_set:
            if (y[:, -1] == -100).sum() != y.size()[0]:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():  # disable gradient calculation
                    pred = compiled_model(x)
                    # print(torch.isnan(pred).sum())
                    pred = pred.reshape(-1, len(config['labels']))
                    mse_loss = criterion(pred, y).mean()
                    # mse_loss=weight_lossf(mse_loss)
                    total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
                    dv_list.append(np.concatenate([y.cpu().numpy(), pred.cpu().numpy()], axis=1))
                    
        dv_list = np.concatenate(dv_list, axis=0)
        dv_mse = total_loss / dv_list.shape[0]  # compute averaged loss   
        # 获取不含-100的行的索引
        dv_r2 = [r2_score(dv_list[np.argwhere(dv_list[:, i] != -100), i], dv_list[np.argwhere(dv_list[:, i] != -100), i + len(label)]) for i in range(len(label))]

        # Save model if your model improved
        if dv_mse < min_mse:
            min_mse = dv_mse
            print('Saving best model available for {}(epoch = {:4d}, lr = {:.4f},  loss = {:.4f}, val_loss = {:.4f}, r2 = {}, val_r2 = {})'.format(label, epoch + 1, scheduler.get_last_lr()[0], tr_mse, min_mse, tr_r2, dv_r2))
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, config['model_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        # visualization loss by tensorboard
        writer.add_scalar('Traing MSE', tr_mse, epoch)
        for i in range(len(label)):
            writer.add_scalar(label[i] + ' Traing R2', tr_r2[i], epoch)
        writer.add_scalar('Testing MSE', dv_mse, epoch)
        for i in range(len(label)):
            writer.add_scalar(label[i] + ' Testing R2', dv_r2[i], epoch)
        writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch)

        # add module weight into Tensorboard
        # writer.add_histogram(tag='layer', values=model.layer.weight,global_step=epoch)
        epoch += 1
        loss_record['dev'].append(dv_mse)

        if early_stop_cnt > config['early_stop']:
            # stop training if your model stop improving for 'config['early_stop']' epochs
            break
        del tr_list, dv_list, mse_loss, pred,
        gc.collect()
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

