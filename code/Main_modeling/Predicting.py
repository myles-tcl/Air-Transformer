from Utils_model import Air_Transformer_self
from Utils_predict import Load_model, Prediction_AiT
from Utils_config import config1 as config
import pandas as pd
import argparse
import ast
import gc
import os
# os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == '__main__':
    # pass in the parameters of prediction
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_time',
                        default="2020-10-12",
                        type=str,
                        help='start time for prediction')
    parser.add_argument('--end_time',
                        default=None,
                        type=str, help='end time for prediction')
    parser.add_argument('--batch_size',
                        default=2048,
                        type=int,
                        help='None: using batch size of config file. if you want using larger batch size for prediction to accelerated calculation')
    parser.add_argument('--model',
                        default=Air_Transformer_self,
                        type=None,
                        help='deep learning model')
    parser.add_argument('--model_name',
                        default='AiT_{}_{}_cv{}'.format(config['res'], config['split'], str(config['cv'])),
                        type=str,
                        help='model name for save')
    parser.add_argument('--device',
                        default=config['device'],
                        type=str,
                        help='None: using batch size of config file. if you want using larger batch size for prediction to accelerated calculation')
    parser.add_argument('--device_ids',
                        default=config['device_ids'],
                        type=ast.literal_eval,
                        help='None: using batch size of config file. if you want using larger batch size for prediction to accelerated calculation')
    args = parser.parse_args()
    # 加载labels
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.start_time is not None:
        config['start_time'] = args.start_time
    if args.end_time is not None:
        config['end_time'] = args.end_time
    config['model'] = args.model
    config['model_name'] = args.model_name
    config['device'] = args.device
    config['device_ids'] = args.device_ids
    # load model
    model = Load_model(config)
    # predicting
    for date in pd.date_range(config['start_time'], config['end_time'], freq=config['t_scale']):
        result = Prediction_AiT(config, date, model, land=config['land'])
        del result
        gc.collect()
        print("Prediction is Fininshed for date: {}".format(date))

