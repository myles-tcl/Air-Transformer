import torch
import argparse
import json
import ast


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_time',
                        default='2020-01-08',
                        type=str,
                        help='start time for prediction')
    parser.add_argument('--end_time',
                        default='2020-12-31',
                        type=str, help='end time for prediction')
    # global parameters
    parser.add_argument('--res', default='0.005', type=str, help='Spatial resolution of input raster data (.nc)')
    parser.add_argument('--loaddata_mode',
                        default='Custom',
                        type=str,
                        help='Decide the method of loading nc, ocean file. Custom: customization in dataloader file. None: pass nc path by args')
    # parser.add_argument('--nc_path',
    #                     default=[['', {'time': True}], ['', {'time': False}]],
    #                     type=ast.literal_eval,
    #                     help='Path of input data, which supporting multi nc files if time dimension is not existed.\
    #                           Each value is a list contains file path and time dict. The dict provide the information of time dimension')
    parser.add_argument('--land',
                        default=False,
                        type=bool,
                        help='decide whether mask the ocean for predicting. if True, the ocean will be masked')
    parser.add_argument('--ocean_path',
                        default='../../data/Auxiliary_data/mask_{}'.format(parser.parse_args().res),
                        type=str,
                        help='filepath of oceam mask nc File, the land areas is Nan')
    parser.add_argument('--engine',
                        default='netcdf4',
                        type=str,
                        help='Engine to use when reading files, see xarray.open_dataset(engine=?). Spporting pass multi-nc file by ./*.nc')
    parser.add_argument('--out_path', default='../../', type=str, help='path of output data and model')
    parser.add_argument('--t_scale', default='D', type=str, help='time scale of input data such as D(daily), M(monthly),')
    parser.add_argument('--cv', default=0, type=int,
                        help='The number of cross-validation determined random state of data split')
    parser.add_argument('--wid_s', default=5, type=int, help='spatial window size')
    parser.add_argument('--wid_t', default=7, type=int,
                        help='time steps if input data, 0: inputing this time data, 1: including previously one day"s data')
    # parser.add_argument('--labels', default=['O3', 'NO2', 'PM2.5', 'HCHO', 'O3_GEO', 'NO2_GEO', 'PM25_RH35_GCC_GEO', 'HCHO_GEO',], type=ast.literal_eval, help='target labels for prediction')
    parser.add_argument('--labels', default=['o3_8h', 'no2', 'pm2.5',], type=ast.literal_eval, help='target labels for prediction')
    parser.add_argument('--split', default='random', type=str, help='data split method')
    parser.add_argument('--n_epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--optimizer', default='AdamW', type=str, help='optimiter')
    parser.add_argument('--optim_hparas', default={'lr': 0.0005}, type=dict, help='hyperparameters for optimizer')
    parser.add_argument('--warmup_steps', default=50, type=int, help='number of warm-up steps for optimizer')
    parser.add_argument('--early_stop', default=200, type=int, help='number of steps without improvement before early stopping')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str, help='ID of GPU to use')
    parser.add_argument('--device_ids', default=[0], type=ast.literal_eval, help='IDs of GPUs to use for multi-GPU')
    parser.add_argument('--feature_name',
                        default=('AOD',
                                'AODANA',
                                'AODINC',
                                'AOD_cwv',
                                'EVI',
                                'Gap_Filled_DNB_BRDF_Corrected_NTL',
                                'LST_Day_1km',
                                'NDVI',
                                'NO2_slant_column_number_density',
                                'O3_column_number_density',
                                'absorbing_aerosol_index',
                                'bld',
                                'blh',
                                'cbh',
                                'cloud_fraction',
                                'cvl',
                                'd2m',
                                'e',
                                'fal',
                                'ilspf',
                                'mcc',
                                'mer',
                                'msdwlwrfcs',
                                'msdwswrfcs',
                                'msl',
                                'sp',
                                't2m',
                                'tco3',
                                'tcw',
                                'tp',
                                'tropospheric_NO2_column_number_density',
                                'u10',
                                'v10',
                                'NOx_gau',
                                'SO2_gau',
                                'PM_gau',
                                'dem',
                                'people_density',
                                'road_gau',),
                        type=tuple,
                        help='name of input features')
    parser.add_argument('--num_workers', default=16, type=int, help='the number of cpu for Dataloader')

    # mdoel parameters
    parser.add_argument('--attn_depth', default=(1, 3, 2, 1), type=ast.literal_eval,
                        help='the numbers of Transformer block (lens of tuple) and the stack layers of each Transformer block (values for each key)')
    parser.add_argument('--patch_size', default=(2, 2), type=ast.literal_eval,
                        help='the size of the convolution kernel of the embedding block')
    parser.add_argument('--embed_dim', default=(None, 64, 96, 128), type=ast.literal_eval,
                        help='the number of channels of each Transformer block. if patchembed[0] == False, embed_dim == the number of variables (input data + spatialtemporal covariates added by the model itself)')
    parser.add_argument('--patchembed', default=(False, True, True, True), type=ast.literal_eval,
                        help='decide whether the number of channels is changed when the data enters the next Transformer blocks')
    parser.add_argument('--std', default=(False, True, True, True), type=ast.literal_eval,
                        help='decide whether to reduce a data frame when the patchembed is operated')
    parser.add_argument('--num_heads', default=(1, 1, 1, 1), type=ast.literal_eval,
                        help='the numner of heads of spatio-temporal self-attention')
    parser.add_argument('--frame', default=(8, 4, 2, 1), type=ast.literal_eval,
                        help='the numner of frames of feature in each Transformer block')
    parser.add_argument('--channel_attention', default=(True, True, True, True,), type=ast.literal_eval,
                        help='decide whether to process the variabels self-attention for each Transformer block')
    parser.add_argument('--drop', default=(0.05, 0.05, 0.05), type=ast.literal_eval,
                        help='the drop rate of the MPL, self-attention and DropPath')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    config_global = {
        'start_time': args.start_time,
        'end_time': args.end_time,
        'loaddata_mode': args.loaddata_mode,
        # 'nc_path': args.nc_path,
        'land': args.land,
        'ocean_path': args.ocean_path,
        'engine': args.engine,
        'num_workers': args.num_workers,
        'out_path': args.out_path,
        'res': args.res,
        'cv': args.cv,
        'wid_s': args.wid_s,
        'wid_t': args.wid_t,
        't_scale': args.t_scale,
        'split': args.split,
        'labels': args.labels,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'optim_hparas': args.optim_hparas,
        'warmup_steps': args.warmup_steps,
        'early_stop': args.early_stop,
        'device': args.device,
        'device_ids': args.device_ids,
        'feature_name': args.feature_name
    }

    config_model = {
        'attn_depth': args.attn_depth,
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'patchembed': args.patchembed,
        'std': args.std,
        'num_heads': args.num_heads,
        'frame': args.frame,
        'channel_attention': args.channel_attention,
        'drop': args.drop
    }

    with open('config_global.json', 'w') as f:
        json.dump(config_global, f)
    with open('config_model.json', 'w') as f:
        json.dump(config_model, f)

