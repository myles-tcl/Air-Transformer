import xarray as xr
from Utils_config import config1 as config
import pandas as pd
import numpy as np
import os
import torch
from Dataloader import time_process
import math

def year_func(ds):
    return ds.expand_dims('time')

def merging_func(ds):
    return ds

def Calcualte_Norm(config):
    engine = config['engine'] 
    file_type = 'zarr' if engine == 'zarr' else 'nc'
    out_path, t_scale, res = config['out_path'], config['t_scale'], config['res']
    # 加载非时序数据
    ds_noseries = xr.open_dataset(f'{out_path}data/Auxiliary_data/Auxiliary_{res}.{file_type}', engine=engine)
    DS = xr.open_mfdataset(f'{out_path}data/Merging_data_{t_scale}/Merging_{res}_*.{file_type}', engine=engine, chunks='auto', parallel=True,
                           concat_dim='time', combine='nested', preprocess=merging_func,)
    ds_year = xr.open_mfdataset(f'{out_path}data/Auxiliary_data/Annually/Annually_*_{res}.{file_type}', engine=engine,
                                # preprocess=year_func, concat_dim='time', combine='nested'
                                )
    ocean = xr.open_dataset(config['ocean_path'] + f'.{file_type}', engine=engine)
    # 将海洋区域全部设置为np.nan，以减少因为海洋区域太多0导致计算和均值偏小
    # 计算输入变量的归一化参数
    DS_mean = [ds.where(ocean['ocean'].notnull(), np.nan).mean(skipna=True).to_array().values for ds in [DS, ds_noseries, ds_year]]
    print('The calculation of Mean values is finished')
    DS_std = [ds.where(ocean['ocean'].notnull(), np.nan).std(skipna=True).to_array().values for ds in [DS, ds_noseries, ds_year]]
    print('The calculation of Std values is finished')
    # DS_mean = [ds.mean(skipna=True).to_array().values for ds in [DS, ds_noseries, ds_year]]
    # DS_std = [ds.std(skipna=True).to_array().values for ds in [DS, ds_noseries, ds_year]]
    var_name = list(DS.data_vars) + list(ds_noseries.data_vars) + list(ds_year.data_vars)
    # 计算空间变量的归一化参数
    lon, lat = DS.x.values, DS.y.values
    xx, yy = np.meshgrid(lon, lat)
    s1 = np.cos(2 * math.pi * xx / 180) * np.cos(2 * math.pi * yy / 90)  # numpy
    s2 = np.cos(2 * math.pi * xx / 180) * np.sin(2 * math.pi * yy / 90)
    s3 = np.sin(2 * math.pi * xx / 180)
    c_mean = np.array([lon.mean(), lat.mean(), s1.mean(), s2.mean(), s3.mean()])
    c_std = np.array([lon.std(), lat.std(), s1.std(), s2.std(), s3.std()])
    # 计算时间变量的归一化参数
    t_scale = config['t_scale']
    date_range = pd.date_range(config['start_time'], config['end_time'], freq=t_scale)
    tp = time_process(date_range, wid_t=1)
    if t_scale == 'M':
        time_list = np.array(tp.d_to_trend())
    if t_scale == 'D':
        time_list = np.array(tp.d_to_trend(t_scale=t_scale) + tp.d_to_jdn() + tp.d_to_year_week())
    t_mean, t_std = time_list.mean(axis=1), time_list.std(axis=1)
    print('The calculation of time and spatial values is finished')

    # 存储输入变量的均值和方差
    df = pd.DataFrame([np.concatenate(DS_mean), np.concatenate(DS_std)], index=['mean', 'std'], columns=var_name)
    save_path = config['out_path'] + 'data/Normalize_{}/'.format(config['t_scale'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.T.to_csv(save_path + 'Standard_Features_{}.csv'.format(config['res']))
    # 存储时空变量的均值和方差
    torch.save({'mean': torch.FloatTensor(np.concatenate([c_mean, t_mean])), 'std': torch.FloatTensor(np.concatenate([c_std, t_std]))}, save_path + 'Standard_Spatiotemporal_{}.pt'.format(config['res']))
    return None


def Load_Normalization_AiT(config):
    path = config['out_path']
    res = config['res']
    feature_name = config['feature_name']
    labels = config['labels']
    t_scale = config['t_scale']
    # load features normalized parameters
    norm_path = '{}data/Normalize_{}/'.format(path, t_scale)
    feas_norm = pd.read_csv(norm_path + 'Standard_Features_{}.csv'.format(res), index_col=0)
    feas_mean, feas_std = torch.FloatTensor(feas_norm.loc[feature_name, 'mean'].values), torch.FloatTensor(feas_norm.loc[feature_name, 'std'].values)

    # load labels mean and std
    y_mean, y_std = torch.FloatTensor(feas_norm.loc[labels, 'mean'].values), torch.FloatTensor(feas_norm.loc[labels, 'std'].values)
    # load spatitemporal mean and std
    tp_norm = torch.load(norm_path + 'Standard_Spatiotemporal_{}.pt'.format(res))
    tp_mean, tp_std = tp_norm['mean'], tp_norm['std']
    # for the data with spatialtemporal var std values is 0 induced by short date range, make the mean and std to 0 and 1
    tp_mean[tp_std == 0], tp_std[tp_std == 0] = 0, 1
    # concat mean and std of all input data
    x_mean, x_std = torch.concat([tp_mean, feas_mean]), torch.concat([tp_std, feas_std])
    # print('The shape of Mean and std for x, y is: ', x_mean.shape, x_std.shape, y_mean.shape, y_std.shape)
    return x_mean, x_std, y_mean, y_std


if __name__ == '__main__':
    Calcualte_Norm(config)
