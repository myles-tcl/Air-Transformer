import xarray as xr
import numpy as np
import torch
import math
import pandas as pd
# from numba import njit
from torch.utils.data import Dataset, DataLoader
# from prefetch_generator import BackgroundGenerator
import warnings


def Dataset_to_tensor(
    DS,
    feature_name=None,
):
    # 将dataset中的指定变量转化为Tensor
    len_time = DS.time.values.shape[0]
    feature_name = list(DS.data_vars) if feature_name is None else feature_name
    ds_feature = [torch.FloatTensor(DS[var].values.copy()) if DS[var].shape[0] == len_time else torch.FloatTensor(DS[var].values.copy()).unsqueeze(0).repeat(len_time, 1, 1) for var in feature_name]
    return ds_feature


"""添加时间变量"""
# Julian day
class time_process():
    def __init__(self, t, wid_t):
        if wid_t != 0:
            self.year, self.month, self.weekdays, self.day = t.year.values, t.month.values, t.day_of_week.values, t.day_of_year.values
        else:
            self.year, self.month, self.weekdays, self.day = t.year, t.month, t.day_of_week, t.day_of_year

    def d_to_jdn(self, ):
        a = (14 - self.month) / 12
        y = self.year + 4800 - a
        m = self.month + 12 * a - 3
        JDN = self.day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045
        return [JDN]

    def d_to_trend(self, t_scale='M'):
        if t_scale == 'D':
            sin_sea = np.sin((2 * math.pi * self.day) / 365.24)
            cos_sea = np.cos((2 * math.pi * self.day) / 365.24)
            sin_mon = np.sin((2 * math.pi * self.month) / 12)
            cos_mon = np.cos((2 * math.pi * self.month) / 12)
            return [sin_sea, cos_sea, sin_mon, cos_mon]
        if t_scale == 'M':
            sin_mon = np.sin((2 * math.pi * self.month) / 12)
            cos_mon = np.cos((2 * math.pi * self.month) / 12)
            return [sin_mon, cos_mon]

    def d_to_year_week(self, ):
        return [self.year, self.weekdays]


# 时间处理为二维，覆盖所有时间和样本
def time_main(time, wid_t=0, wid_s=5, t_scale='M'):
    if wid_t != 0:
        time = pd.date_range(time - pd.Timedelta(wid_t, t_scale), time, freq=t_scale)
    # 时间从年月日变换为Julian day，季节和月趋势
    tp = time_process(time, wid_t)
    if t_scale == 'M':
        time_list = np.array(tp.d_to_trend())
    if t_scale == 'D':
        time_list = np.array(tp.d_to_trend(t_scale=t_scale) + tp.d_to_jdn() + tp.d_to_year_week())

    # 维度扩充，当前样本当前时间下10*10重复填充
    var_t = time_list.reshape(-1, wid_t + 1, 1, 1,).repeat(wid_s, axis=2).repeat(wid_s, axis=3)
    return var_t


# @njit()
def cal_coords(lon, lat):
    s1 = np.cos(2 * np.pi * lon / 180) * np.cos(2 * np.pi * lat / 90)  # numpy
    s2 = np.cos(2 * np.pi * lon / 180) * np.sin(2 * np.pi * lat / 90)
    s3 = np.sin(2 * np.pi * lon / 180)
    coords_var = np.array([lon, lat, s1, s2, s3])
    return coords_var


### 完整数据集中添加回来时间变量
def Loading_ponits_with_neighborgrid(ds_feature, ds_target, longitude, latitude, x, y, config, var_t):
    # 提取具有领域信息的特征数据
    data = ds_feature[:, :, y:y + config['wid_s'], x:x + config['wid_s']]  # numpy
    # 添加空间变量
    lon, lat = longitude[x], latitude[y]
    coords_var = cal_coords(lon, lat)
    coords_var = coords_var.reshape(-1, 1, 1, 1).repeat(config['wid_t'] + 1, axis=1).repeat(config['wid_s'], axis=2).repeat(config['wid_s'], axis=3)
    # 合并
    data = np.concatenate([coords_var, var_t, data], axis=0)
    # 判断是否存在缺失值，若缺失则直接填补
    if np.isnan(data).sum() > 0:
        data = np.nan_to_num(data, 0)
    # 提取标签，依次是labels, longitude, latitude
    target = ds_target[:, -1, y, x]
    target = np.concatenate([target, np.array([lon, lat])])
    return data, target


class Mydataset_neighborhood(Dataset):
    def __init__(self, DS, idx_surface, config, mode=None):
        self.idx_surface = idx_surface
        self.config = config
        self.mode = mode
        # load feature
        ds_feature = torch.stack(Dataset_to_tensor(DS.drop(config['labels']), tuple(config['feature_name'])))
        print('Loading Feature is finished')
        L_pad = config['wid_s'] // 2
        ds_feature = torch.nn.functional.pad(ds_feature, (L_pad, L_pad, L_pad, L_pad), mode='constant', value=0)
        self.ds_feature = ds_feature.numpy()
        # load target
        self.ds_target = torch.stack(Dataset_to_tensor(DS, config['labels'])).numpy()
        # to coords and time
        self.longitude = DS.x.values
        self.latitude = DS.y.values
        xx, yy = np.meshgrid(self.longitude, self.latitude)
        self.time = pd.to_datetime(DS.isel(time=-1)['time'].values)
        self.var_t = time_main(self.time, config['wid_t'], config['wid_s'], t_scale=config['t_scale'])
        print('Loading Targets is finished')

    def __getitem__(self, i):
        y, x = self.idx_surface[i]

        data, target = Loading_ponits_with_neighborgrid(self.ds_feature, self.ds_target,
                                                        self.longitude, self.latitude, x, y,
                                                        self.config, self.var_t)

        data = torch.FloatTensor(data)
        target = torch.DoubleTensor(target)

        return data, target

    def __len__(self):
        return self.idx_surface.shape[0]


# # 在后台提前加载下一个batch的数据
# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())


def Mydataloader_neighborhood(
    DS,
    idx_surface,
    config,
    mode=None,
):
    dataset = Mydataset_neighborhood(DS, idx_surface, config, mode)
    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=False,
                            prefetch_factor=8)
    return dataloader


def Loading_dataloader(
    date,
    config,
    mode='Training',
    land=False,
):
    # 确定时间步长
    date_range = slice(np.datetime64(date, config['t_scale']) - np.timedelta64(config['wid_t'], config['t_scale']), np.datetime64(date, config['t_scale']))
    # 自定义加载nc数据集
    engine = config['engine']
    file_type = 'zarr' if engine == 'zarr' else 'nc'
    out_path, t_scale, res = config['out_path'], config['t_scale'], config['res']
    # 加载时序数据monthly
    ds_noseries = xr.open_dataset(f'{out_path}data/Auxiliary_data/Auxiliary_{res}.{file_type}', engine=engine)
    DS = xr.open_mfdataset(f'{out_path}data/Merging_data_{t_scale}/Merging_{res}_*.{file_type}', engine=engine, 
                           concat_dim='time', combine='nested',)
    ds_year = xr.open_dataset(f'{out_path}data/Auxiliary_data/Annually/Annually_{str(date.year)}_{res}.{file_type}', engine=engine)
    if (DS.y.values == ds_noseries.y.values).sum() == 0:
        ds_noseries['y'] = DS.y
        
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        DS = xr.merge([DS, ds_noseries, ds_year]).sel(time=date_range)

    # 按需求提取dataloader要加载的数据x,y索引
    if mode == 'Training':
        # index first label with vlaues to build training dataset
        conditions = [DS.sel(time=date)[label].values >= 0 for label in config['labels']]
        idx_surface = np.argwhere(np.any(conditions, axis=0))
        # idx_surface = np.argwhere((DS.sel(time=date)[config['labels'][0]].values >= 0) | (DS.sel(time=date)[config['labels'][1]].values >= 0) | (DS.sel(time=date)[config['labels'][2]].values >= 0) | (DS.sel(time=date)[config['labels'][3]].values >= 0))
        print(idx_surface.shape)
        
    if mode == 'Predicting':
        if land:
            ocean = xr.open_dataset(config['ocean_path'] + f'.{file_type}', engine=engine)
            idx_surface = np.argwhere(ocean['ocean'].notnull().values)
        else:
            idx_surface = np.argwhere(DS.sel(time=date)['u10'].notnull().values)
    # load data by dataloader
    Dataloader_Extrating_Traing = Mydataloader_neighborhood(DS, idx_surface, config, mode=mode)

    del DS, ds_noseries, ds_year
    return Dataloader_Extrating_Traing
