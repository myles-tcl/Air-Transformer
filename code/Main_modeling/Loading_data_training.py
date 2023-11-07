from Dataloader import Loading_dataloader
import pandas as pd
import numpy as np
import os
import torch
import argparse
from Utils_config import config1 as config

if __name__ == '__main__':

    # pass in the parameters of prediction
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_time',
                        default=None,
                        type=str,
                        help='starting time of loading traning dataset')
    parser.add_argument('--end_time',
                        default=None,
                        type=str,
                        help='end time of loading traning dataset')
    args = parser.parse_args()
    if args.start_time is not None:
        config['start_time'] = args.start_time
    if args.end_time is not None:
        config['end_time'] = args.end_time

    # 加载labels
    save_path = '{}data/Training_{}/{}*{}_{}/'.format(config['out_path'], config['t_scale'], config['wid_s'], config['wid_t'], str(config['res']))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_all, target_all = [], []
    for date in pd.date_range(config['start_time'], config['end_time'], freq=config['t_scale']):
        # load training dataset by Dataloader
        print(f"Strating Load Training Dataset From Zarr for date: {date}")
        if not os.path.exists(save_path + 'feature_{}.pt'.format(str(date.date()))):
            Dataloader_Extrating_Training = Loading_dataloader(
                date,
                config,)
            
            results = [[data, target] for data, target in Dataloader_Extrating_Training]
            try: 
                data_list, target_list = zip(*results)
                if len(data_list) > 0:
                    data_list = torch.cat(data_list, dim=0)
                    target_list = torch.cat(target_list, dim=0)
                    print("Loading Training Dataset From Zarr is Fininshed for date: {}, the shape of feature is {}".format(date, data_list.shape))
                    print('The number of nan values is: Feature({}), Target({})'.format(torch.isnan(data_list).sum(), torch.isnan(target_list).sum()))
                    torch.save(torch.nan_to_num(data_list, 0), save_path + 'feature_{}.pt'.format(str(date.date())))
                    torch.save(target_list, save_path + 'target_{}.pt'.format(str(date.date())))
            except Exception as e:
                print(e)
    print('Loading data from dataset is fininshed')
    data_all, target_all = [], []
    for date in pd.date_range(config['start_time'], config['end_time'], freq=config['t_scale']):
        try:
            data_all.append(torch.load(save_path + 'feature_{}.pt'.format(str(date.date()))))
            target_all.append(torch.load(save_path + 'target_{}.pt'.format(str(date.date()))))
        except Exception as e:
            print(e)

    data_all, target_all = torch.cat(data_all, dim=0), torch.cat(target_all, dim=0)
    print('The Shape of Features is : {} , Target is : {}'.format(data_all.shape, target_all.shape))

    torch.save(data_all, save_path + 'Feature.pt')
    torch.save(target_all, save_path + 'Target.pt')
