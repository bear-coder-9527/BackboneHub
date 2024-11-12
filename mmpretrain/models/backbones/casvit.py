
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import einsum
from torch.nn import functional as F
from itertools import chain
from einops import rearrange
from itertools import chain

from mmcv.cnn import Linear
from mmcv.cnn.bricks import (build_activation_layer, build_conv_layer,
                            build_norm_layer, ConvModule, DropPath,
                            Dropout)
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.cnn.bricks.drop import build_dropout

from mmengine.model import BaseModule, ModuleList, Sequential
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS
from mmpretrain.models.utils.helpers import to_2tuple


class ConvStem(BaseModule):
    """ 
    Image to Patch Embedding with Convolutional Layers.

    This module converts an input image into embedded patches through 
    a sequence of two convolutional layers with downsampling.

    Args:
        in_channels (int): Number of input channels, typically 3 for RGB images. Default: 3.
        out_channels (int): Number of output channels after embedding. Default: 64.
        conv_cfg (dict, optional): Configuration dictionary for convolution layers. Default: None.
        norm_cfg (dict, optional): Configuration dictionary for normalization layers. Default: BatchNorm.
        act_cfg (dict, optional): Configuration dictionary for activation layers. Default: ReLU.
        init_cfg (dict, optional): Initialization configuration dictionary. Default: None.
    """
   
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 64,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ) -> None:
        super(ConvStem, self).__init__(init_cfg=init_cfg)
        mid_channels = out_channels // 2

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3, 
            stride=2,
            padding=1,
            bias=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to transform the input image into patch embeddings.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor with embedded patches 
                    of shape (B, out_channels, H/4, W/4).
        """
        
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class PatchEmbedLayer(BaseModule):
    """Image Patch Embedding Layer.

    This module uses a convolution layer to embed image patches.

    Args:
        patch_size (int): Size of each image patch. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embedding_dims (int): Dimension of the embedding output. Default: 768.
        stride (int): Convolution stride. Default: 16.
        padding (int): Convolution padding. Default: 0.
        conv_cfg (dict, optional): Configuration for convolution layer. 
                Default: None.
        norm_cfg (dict, optional): Configuration for normalization layer. 
                Default: BatchNorm.
        init_cfg (dict, optional): Initialization config. Default: None.
    """
    def __init__(self,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embedding_dims: int = 768,
                 stride: int = 16,
                 padding: int = 0,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 init_cfg: Optional[dict] = None
                 ) -> None:
        super(PatchEmbedLayer, self).__init__(init_cfg=init_cfg)
        
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)

        self.proj = build_conv_layer(
            cfg=conv_cfg,
            in_channels=in_channels,
            out_channels=embedding_dims,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = build_norm_layer(norm_cfg, embedding_dims)[1] \
            if norm_cfg else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor after convolution and normalization.
        """
        out = self.proj(x)
        out = self.norm(out)
        return out
    
class Mlp(BaseModule):
    """Multilayer Perceptron (MLP) with optional convolution layers.

    Args:
        in_features (int): Dimension of input features.
        hidden_features (int, optional): Dimension of hidden layer features. 
                        Defaults to `in_features`.
        out_features (int, optional): Dimension of output features. 
                    Defaults to `in_features`.
        drop (float): Dropout rate. Default: 0.0.
        conv_cfg (dict, optional): Configuration for convolution layers. 
                    Default: None.
        act_cfg (dict, optional): Configuration for activation layer. 
                    Default: GELU.
        init_cfg (dict, optional): Initialization config. 
                    Default: None.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop: float = 0.,
                 conv_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ) -> None:
        super(Mlp, self).__init__(init_cfg=init_cfg)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = build_conv_layer(
            cfg=conv_cfg,
            in_channels=in_features, 
            out_channels=hidden_features,
            kernel_size=1
        )
        self.act = build_activation_layer(act_cfg)
        self.fc2 = build_conv_layer(
            cfg=conv_cfg,
            in_channels=hidden_features, 
            out_channels=out_features,
            kernel_size=1
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor after MLP transformations.
        """
        out = self.fc1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)

        return out

class SpatialOperation(BaseModule):
    """Spatial Attention Operation.

    This module applies a spatial attention operation using depth-wise
    and point-wise convolutions.

    Args:
        dim (int): Dimension of the input features.
        conv_cfg (dict, optional): Configuration for convolution layer. 
                Default: None.
        norm_cfg (dict, optional): Configuration for normalization layer. 
                Default: BatchNorm.
        act_cfg (dict, optional): Config for activation functions. 
               Default: ReLU and Sigmoid.
        init_cfg (dict, optional): Initialization config. 
                Default: None.
    """
    def __init__(self,
                 dim: int = None,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = (dict(type='ReLU'),
                                            dict(type='Sigmoid')),
                 init_cfg: Optional[dict] = None
                 ) -> None:
        super(SpatialOperation, self).__init__(init_cfg=init_cfg)
        self.block = Sequential(
            ConvModule(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim,
                bias=True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg[0]
            ),
            ConvModule(
                in_channels=dim,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=act_cfg[1]
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatial attention.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying spatial attention.
        """
        out = x * self.block(x)
        return out
    
class ChannelOperation(BaseModule):
    """Channel Attention Operation with Avg Pooling.

    Args:
        dim (int): Dimension of the input features.
        conv_cfg (dict, optional): Configuration for convolution layer. Default: None.
        norm_cfg (dict, optional): Normalization config for convolution. Default: BatchNorm.
        act_cfg (dict, optional): Activation function config. Default: Sigmoid.
        init_cfg (dict, optional): Initialization config. Default: None.
    """
    def __init__(self,
                 dim: int = None,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='Sigmoid'),
                 init_cfg: Optional[dict] = None
                 ) -> None:
        super(ChannelOperation, self).__init__(init_cfg=init_cfg)
        self.block = Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            ConvModule(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=act_cfg),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for channel attention.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with applied channel attention.
        """
        out = x * self.block(x)
        return out

class LocalIntegration(BaseModule):
    """Local Integration Layer.

    Integrates local information with convolution layers.

    Args:
        dim (int): Dimension of the input features.
        ratio (int): Scaling ratio for hidden layer size. Default: 1.
        conv_cfg (dict, optional): Convolution config. Default: None.
        norm_cfg (dict, optional): Normalization config. Default: GELU.
        act_cfg (dict, optional): Activation config. Default: ReLU.
        init_cfg (dict, optional): Initialization config. Default: None.
    """
    def __init__(self,
                 dim: int = None,
                 ratio: int = 1,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='GELU'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ) -> None:
        super(LocalIntegration, self).__init__(init_cfg=init_cfg)
        
        mid_channels = round(ratio * dim)
        self.network = Sequential(
            ConvModule(
                in_channels=dim,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None
            ),
            ConvModule(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=mid_channels,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=act_cfg
            ),
            build_conv_layer(
                cfg=conv_cfg,
                in_channels=mid_channels,
                out_channels=dim,
                kernel_size=1,
                stride=1,
                padding=0
            )
            
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        return out
    
class AdditiveTokenMixer(BaseModule):
    """
    A token mixing module that combines spatial and channel operations 
    to integrate token representations with additive attention mechanism.

    Args:
        dim (int): The embedding dimension of input tokens. 
            Default: 512.
        attn_bias (bool): If True, enables bias for attention layers. 
            Default: False.
        proj_drop (float): Dropout rate for the final projection layer. 
            Default: 0.0.
        conv_cfg (dict, optional): Configuration for convolutional layers.
            Used for setting layer types, kernels, strides, etc. Default: None.
        init_cfg (dict, optional): Initialization configuration for layers.
            Specifies the initialization methods and target layers. 
            Default: None.
    """
    def __init__(self,
                 dim: int = 512,
                 attn_bias: bool = False,
                 proj_drop: float = 0.,
                 conv_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None
                 ) -> None:
        super(AdditiveTokenMixer, self).__init__(init_cfg=init_cfg)
        self.qkv = build_conv_layer(
            cfg=conv_cfg,
            in_channels=dim,
            out_channels=dim * 3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=attn_bias
        )
        self.oper_q = Sequential(
            SpatialOperation(dim=dim),
            ChannelOperation(dim=dim)
        )
        self.oper_k = Sequential(
            SpatialOperation(dim=dim),
            ChannelOperation(dim=dim)
        )
        self.dwc = build_conv_layer(
            cfg=conv_cfg,
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim
        )
        
        self.proj =build_conv_layer(
            cfg=conv_cfg,
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim
        )
        self.proj_drop = Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to process input tensor through additive token mixing.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Output tensor after additive token mixing, 
                    with the same shape as input.
        """

        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        
        return out

class AdditiveBlock(BaseModule):
    """Additive Block.

    This module implements an additive block with local perception,
    attention-based token mixing, and a multi-layer perceptron (MLP) layer.

    Args:
        dim (int): The dimensionality of input features. Default: None.
        mlp_ratio (float): The ratio of hidden dimension size to input
            dimension size in the MLP layer. Default: 4.0.
        attn_bias (bool): Whether to use bias in the attention layer. Default: False.
        drop (float): Dropout rate for the attention and MLP layers. Default: 0.0.
        drop_path (float): Stochastic depth rate for drop path. Default: 0.0.
        conv_cfg (dict, optional): Configuration dictionary for convolution
            layers, not used in this implementation. Default: None.
        norm_cfg (dict, optional): Configuration dictionary for normalization layers.
            Default is GELU activation.
        act_cfg (dict, optional): Configuration dictionary for activation layers.
            Default is ReLU activation.
        init_cfg (dict, optional): Initialization configuration for the module.
            Default: None.
    """
    def __init__(self,
                 dim: int = None,
                 mlp_ratio: float = 4.,
                 attn_bias: bool = False,
                 drop: float = 0.,
                 drop_path: float = 0.,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='GELU'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ) -> None:
        super(AdditiveBlock, self).__init__(init_cfg=init_cfg)
        self.local_perception = LocalIntegration(
            dim=dim,
            ratio=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = AdditiveTokenMixer(
            dim=dim,
            attn_bias=attn_bias,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Additive Block.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W), where B is
                batch size, C is channels, H is height, and W is width.

        Returns:
            Tensor: Output tensor after additive block operations,
            with the same shape as the input tensor.
        """
        out = x + self.local_perception(x)
        out = out + self.drop_path(self.attn(self.norm1(out)))
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        
        return out
        
@MODELS.register_module()
class CASViT(BaseBackbone):
    """CAS-ViT.

    A PyTorch implement of : 
    `CAS-ViT: Convolutional Additive Self-attention Vision Transformers 
    for Efficient Mobile Applications
    <https://arxiv.org/abs/2408.03703>`_

    Inspiration from
    https://github.com/Tianfang-Zhang/CAS-ViT

    Args:
        arch (str | dict): CAS-ViT architecture. If use string, choose
            from 't', 's', 'xs' and 'm'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.

            Defaults to 't'.
        in_channels (int): Number of input channels, typically set to 3 for RGB images. 
            Default: 3.
        out_indices (int | list[int]): Index of output stages. -1 for the last stage. 
            Default: -1.
        frozen_stages (int): Number of stages to freeze. 0 means no stages are frozen. 
            Default: 0.
        attn_bias (bool): Whether to use bias in attention layers. 
            Default: False.
        mlp_ratio (int): Ratio to adjust the hidden dimensions of MLP layers. 
            Default: 4.
        drop_rate (float): Dropout rate applied after each MLP and attention layer. 
            Default: 0.0.
        drop_path_rate (float): Drop path rate for stochastic depth in each stage. 
            Default: 0.0.
        norm_cfg (dict): Configuration for normalization layers. 
            Default: BatchNorm.
        act_cfg (dict): Configuration for activation layers, typically GELU. 
            Default: GELU.
        init_cfg (list[dict]): Initialization configurations, defining types and layers 
            to initialize.
            Default: Kaiming initialization for Conv2d, and Constant initialization 
            for batch normalization.
    """
    arch_settings = {
        'xs': {
            'layers':[2, 2, 4, 2], 
            'embed_dims': [48, 56, 112, 220], 
        },
        's': {
            'layers': [3, 3, 6, 3], 
            'embed_dims': [48, 64, 128, 256], 
        },
        'm': {
            'layers': [3, 3, 6, 3], 
            'embed_dims': [64, 96, 192, 384], 
        },
        't': {
            'layers': [3, 3, 6, 3], 
            'embed_dims': [96, 128, 256, 512], 
        },
    }  
    def __init__(self,
                 arch = 's',
                 in_channels: int = 3,
                 out_indices=-1,
                 frozen_stages=0,
                 attn_bias: bool = False,
                 mlp_ratio: int = 4,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]) -> None:
        super().__init__(init_cfg=init_cfg)
        
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'   
        
        self.layers = arch['layers']
        self.embed_dims = arch['embed_dims']   
        self.num_stages = len(self.layers)  
        
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.layers))
        ]
        block_idx = 0 
        
        # 4 downsample layers between stages, including the stem layer.               
        self.downsample_layers = ModuleList()
        stem = ConvStem(
            in_channels=in_channels,
            out_channels=self.embed_dims[0]
        )
        self.downsample_layers.append(stem)
        
        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = ModuleList()
        
        for i in range(self.num_stages):
            depth = self.layers[i]
            channels = self.embed_dims[i]
            if i >= 1:
                downsample_layer = PatchEmbedLayer(
                    patch_size=3,
                    stride=2,
                    padding=1,
                    in_channels=self.embed_dims[i - 1],
                    embedding_dims=self.embed_dims[i]
                )
                self.downsample_layers.append(downsample_layer)
            
            stage = Sequential(*[
                AdditiveBlock(
                    dim=channels,
                    mlp_ratio=mlp_ratio,
                    attn_bias=attn_bias,
                    drop=drop_rate,
                    drop_path=dpr[block_idx + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                ) for j in range(depth)
            ])
            block_idx += depth
            
            self.stages.append(stage)
            
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
            assert 0 <= out_indices[i] <= self.num_stages, \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

        if self.out_indices:
            for i_layer in self.out_indices:
                layer = build_norm_layer(norm_cfg, self.embed_dims[i_layer])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        self.frozen_stages = frozen_stages
        self._freeze_stages()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                outs.append(out.flatten(2).mean(-1))
        
        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False
    
    def train(self, mode=True):
        super(CASViT, self).train(mode)
        self._freeze_stages()