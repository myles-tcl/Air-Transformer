import torch
import numpy as np
from sklearn.model_selection import KFold
from Utils_train import Mydataloader_cv, train
from Utils_model import Air_Transformer_self
from Utils_config import config1 as config
import argparse


def main_train_cv(config, tr_idx, dv_idx, cv, multi_gpu=False):
    # load dataloader
    tr_set, dv_set = Mydataloader_cv(
        config,
        n_jobs=4,
        shuffle=True,
        tr_idx=tr_idx,
        dv_idx=dv_idx,
        cv=cv,
    )
    # multi-gpu
    if multi_gpu:
        config['model'] = torch.nn.DataParallel(config['model'], device_ids=config['device_ids'])
    # Training model
    min_mse, loss_record = train(tr_set, dv_set, config, strict=False)
    return None


if __name__ == '__main__':
    # pass in the parameters of prediction  
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default=Air_Transformer_self,
                        type=None,
                        help='deep learning model')
    parser.add_argument('--suffix',
                        default='_O3NO2PM25',
                        type=str,
                        help='the suffix of the training data set')
    parser.add_argument('--device',
                        default=None,
                        type=str,
                        help='id of gpu')
    parser.add_argument('--PreTraining',
                        default=False,
                        type=str,
                        help='whether to pre-training')
    parser.add_argument('--Fix_encoder',
                        default=False,
                        type=str,
                        help='whether to training encoder')
    parser.add_argument('--Train_labels',
                        default=None,
                        type=list,
                        help='whether to training model using labels pointed')
    args = parser.parse_args()
    if args.device is not None:
        config['device'] = args.device
    config['model'] = args.model
    config['suffix'] = args.suffix
    config['PreTraining'] = args.PreTraining
    config['Fix_encoder'] = args.Fix_encoder
    config['Train_labels'] = args.Train_labels if args.Train_labels is not None else config['labels']
    
    # out-of-samples
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    target = torch.load('{}data/Training_{}/{}*{}_{}/Target{}.pt'.format(config['out_path'], config['t_scale'], config['wid_s'], config['wid_t'], str(config['res']), config['suffix']))
    data_idx = np.array(range(target.shape[0]))
    np.random.seed(100)
    np.random.shuffle(data_idx)

    for cv, (train_idx, test_idx) in enumerate(kf.split(data_idx)):
        if cv == 1:
            config['model_name'] = 'AiT_{}_{}_cv{}'.format(config['res'], config['split'], cv)
            main_train_cv(config, tr_idx=list(train_idx), dv_idx=list(test_idx), cv=cv)
