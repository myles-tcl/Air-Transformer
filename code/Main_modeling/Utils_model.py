from AiT import Air_Transformer
from Utils_config import config1, config2


if config1['t_scale'] == 'M':
    num_st = 5
if config1['t_scale'] == 'D':
    num_st = 10
in_chans = len(config1['feature_name']) + num_st
if not config2['patchembed'][0]:
    config2['embed_dim'][0] = in_chans

Air_Transformer_self = Air_Transformer(image_size=(config1['wid_s'], config1['wid_s']),
                                       in_chans=in_chans,
                                       attn_depth=config2['attn_depth'],
                                       patch_size=config2['patch_size'],
                                       embed_dim=config2['embed_dim'],
                                       patchembed=config2['patchembed'],
                                       std=config2['std'],
                                       num_heads=config2['num_heads'],
                                       num_heads_chans=config2['frame'],
                                       frame=config2['frame'],
                                       channel_attention=config2['channel_attention'],
                                       drop=config2['drop'][0],
                                       attn_drop=config2['drop'][1],
                                       drop_path=config2['drop'][2],
                                       num_regress=len(config1['labels']))
