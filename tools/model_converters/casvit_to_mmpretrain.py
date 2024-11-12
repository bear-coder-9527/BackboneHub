# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict
import os
import mmengine
import torch
from mmengine.runner import CheckpointLoader

def convert_casvit(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        new_v = v
        new_k = k  # 默认情况下，new_k 和原始的 k 相同

        if k.startswith('patch_embed.'):
            new_k = k.replace('patch_embed.', 'backbone.downsample_layers.')
            
            new_k = new_k.replace('backbone.downsample_layers.0', 'backbone.downsample_layers.0.conv1.conv')
        
            new_k = new_k.replace('backbone.downsample_layers.1', 'backbone.downsample_layers.0.conv1.bn')
            
            new_k = new_k.replace('backbone.downsample_layers.3', 'backbone.downsample_layers.0.conv2.conv')
            
            new_k = new_k.replace('backbone.downsample_layers.4', 'backbone.downsample_layers.0.conv2.bn')

        if k.startswith('network.'):
            new_k = k.replace('network.', 'backbone.stages.')
            
            if 'attn.' in new_k:
                new_k = new_k.replace('attn.oper_q.0.block.0', 'attn.oper_q.0.block.0.conv')
                new_k = new_k.replace('attn.oper_q.0.block.1', 'attn.oper_q.0.block.0.bn')
                new_k = new_k.replace('attn.oper_q.0.block.3', 'attn.oper_q.0.block.1.conv')
                new_k = new_k.replace('attn.oper_q.1.block.1', 'attn.oper_q.1.block.1.conv')
                new_k = new_k.replace('attn.oper_k.0.block.0', 'attn.oper_k.0.block.0.conv')
                new_k = new_k.replace('attn.oper_k.0.block.1', 'attn.oper_k.0.block.0.bn')
                new_k = new_k.replace('attn.oper_k.0.block.3', 'attn.oper_k.0.block.1.conv')
                new_k = new_k.replace('attn.oper_k.1.block.1', 'attn.oper_k.1.block.1.conv')
            
            if 'local_perception.' in new_k:
                
                new_k = new_k.replace('local_perception.backbone.stages.0', 'local_perception.network.0.conv')
                new_k = new_k.replace('local_perception.backbone.stages.1', 'local_perception.network.0.bn')
                new_k = new_k.replace('local_perception.backbone.stages.2', 'local_perception.network.1.conv')
                new_k = new_k.replace('local_perception.backbone.stages.4', 'local_perception.network.2')
                
            if 'stages.1' in new_k:
                new_k = new_k.replace('stages.1', 'downsample_layers.1')
            if 'stages.3' in new_k:
                new_k = new_k.replace('stages.3', 'downsample_layers.2')
            if 'stages.5' in new_k:
                new_k = new_k.replace('stages.5', 'downsample_layers.3')
            
            new_k = new_k.replace('stages.2', 'stages.1')
            new_k = new_k.replace('stages.4', 'stages.2')
            new_k = new_k.replace('stages.6', 'stages.3')
            
        if k.startswith('norm'):
            new_k = k.replace('norm', 'backbone.norm3')
        
        if k.startswith('head'):
            new_k = k.replace('head', 'head.fc')

        # 将转换后的键值对存储在新的字典中
        new_ckpt[new_k] = new_v

    return new_ckpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained wtconvnext '
        'models to mmpretrain style.')
    parser.add_argument('infile', type=str, help='Path to the ckpt.')
    parser.add_argument('outfile', type=str, help='Output file.')
    args = parser.parse_args()
    assert args.outfile

    checkpoint = CheckpointLoader.load_checkpoint(args.infile, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert_casvit(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.outfile))
    torch.save(dict(state_dict=weight), args.outfile)

    print('Done!!')