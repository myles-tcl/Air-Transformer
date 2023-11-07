from Dataloader import Loading_dataloader
import pandas as pd
import torch
import time
from tqdm import tqdm
import os
from Calculating_norms import Load_Normalization_AiT
# torch.set_float32_matmul_precision('high')


def Load_model(config, ):
    path = config['out_path']
    t_scale = config['t_scale']
    res = config['res']
    model_name = config['model_name']
    model = config['model']
    device = config['device']

    # load checkpoint
    config['model_path'] = '{}model/{}_{}/{}.pth'.format(path, t_scale, res, model_name)
    checkpoint = torch.load(config['model_path'], map_location='cpu')
    print('load checkpoint from %s' % path + 'model/' + str(res) + '/' + model_name + '.pth')

    if len(config['device_ids']) > 1:
        model = torch.nn.DataParallel(model, device_ids=config['device_ids'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    model.eval()
    # model = torch.compile(model)

    print("Loading model is finised")
    return model


def Prediction_AiT(config, date, model, land=False):
    path = config['out_path']
    t_scale = config['t_scale']
    labels = config['labels']
    len_l = len(config['labels'])
    model_name = config['model_name']
    device = config['device']

    # load normalization params
    config['x_mean'], config['x_std'], config['y_mean'], config['y_std'] = Load_Normalization_AiT(config)
    # load dataloader
    pre_set = Loading_dataloader(
        date,
        config,
        mode='Predicting',
        land=land,
    )
    print('Loading Dataloader is Finished')

    # predicting
    predictions = []
    x_mean = config['x_mean'].reshape(1, -1, 1, 1, 1)
    x_std = config['x_std'].reshape(1, -1, 1, 1, 1)

    t1 = time.time()
    for x, y in tqdm(pre_set):
        # with torch.cuda.stream(torch.cuda.Stream()):
        with torch.no_grad():
            x = (x - x_mean) / x_std
            x = x.to(device)
            pred = model(x)
        y_ = torch.cat([y, pred.detach().cpu().double()], dim=1)
        predictions.append(y_)

    predictions = torch.cat(predictions, axis=0)
    # 反归一化
    predictions[:, -len_l:] = predictions[:, -len_l:] * config['y_std'] + config['y_mean']
    t2 = time.time()
    print('compute spend {} s'.format(t2 - t1))

    pre_path = '{}data/Prediction_{}/{}/'.format(path, t_scale, model_name)
    if not os.path.exists(pre_path):
        os.makedirs(pre_path)
    # to nc
    labels_pre = [i + '_pre' for i in labels]
    df = pd.DataFrame(predictions, columns=labels + ['longitude', 'latitude'] + labels_pre).set_index(['latitude', 'longitude'])
    # df.to_csv(pre_path + 'test.csv')
    df = df.to_xarray().assign_coords(time=date)
    # df = df[labels_pre] * config['y_std'].numpy() + config['y_mean'].numpy()
    # compression_encoding = {label: {"zlib": True, "complevel": 9} for label in labels_pre}
    df.to_netcdf(pre_path + str(date.date()) + '.nc',
                 # engine='h5netcdf', encoding=compression_encoding
                 )
    torch.cuda.empty_cache()
    del pre_set, predictions, df
    return None
