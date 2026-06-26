from typing import List, Tuple, Type, Union

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.swin_unetr_blocks import (
    DropPath,
    LayerNormNd,
    LiteModule,
    LiteSwinEncoder,
    LiteSwinUNETRDecoder,
    SqueezeExcitationNd,
    SwinLiteBlock,
    SwinLiteStage,
    WindowAttention,
    estimate_cost_from_flops,
)
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder


__author__ = ["Stefano Petraccini", "GitHub Copilot"]


class LiteSwinUNETR(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        conv_bias: bool = True,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        stage_depths: Union[int, List[int], Tuple[int, ...], None] = None,
        num_heads: Union[int, List[int], Tuple[int, ...], None] = None,
        window_size: Union[int, List[int], Tuple[int, ...]] = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        encoder_lite_modules_per_stage: Union[int, List[int], Tuple[int, ...], None] = None,
        decoder_lite_modules_per_stage: Union[int, List[int], Tuple[int, ...]] = 1,
        lite_expansion_ratio: float = 2.0,
        lite_se_reduction: int = 4,
    ):
        super().__init__()

        self.key_to_encoder = "encoder"
        self.key_to_stem = "encoder.stem"
        self.keys_to_in_proj = ("encoder.stem",)
        self.key_to_lpe = None

        self.deep_supervision = deep_supervision

        self.encoder = LiteSwinEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            stage_depths=stage_depths,
            num_heads=num_heads,
            window_size=window_size,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            encoder_lite_modules_per_stage=encoder_lite_modules_per_stage,
            lite_expansion_ratio=lite_expansion_ratio,
            lite_se_reduction=lite_se_reduction,
        )

        self.decoder = LiteSwinUNETRDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            lite_modules_per_stage=decoder_lite_modules_per_stage,
            lite_expansion_ratio=lite_expansion_ratio,
            lite_se_reduction=lite_se_reduction,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        if len(input_size) != convert_conv_op_to_dim(self.encoder.conv_op):
            raise AssertionError(
                "just give the image size without color/feature channels or batch channel. "
                "Give input_size=(x, y(, z))"
            )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    def compute_approx_flops(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        if len(input_size) != convert_conv_op_to_dim(self.encoder.conv_op):
            raise AssertionError(
                "just give the image size without color/feature channels or batch channel. "
                "Give input_size=(x, y(, z))"
            )
        return int(self.encoder.compute_approx_flops(input_size) + self.decoder.compute_approx_flops(input_size))

    def estimate_computational_cost(self, input_size: Union[List[int], Tuple[int, ...]], batch_size: int = 1) -> dict:
        return estimate_cost_from_flops(self.compute_approx_flops(input_size), batch_size=batch_size)

    @staticmethod
    def initialize(module: nn.Module):
        if isinstance(
            module,
            (
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            ),
        ):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LayerNorm, LayerNormNd)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)


class SwinUNETR(AbstractDynamicNetworkArchitectures):
    """
    Swin-UNETR style architecture with hierarchical Swin encoder and standard UNet decoder.

    This class is intentionally distinct from LiteSwinUNETR:
    - LiteSwinUNETR uses Lite modules in encoder/decoder stages.
    - SwinUNETR disables Lite modules and uses a convolutional UNetDecoder.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        conv_bias: bool = True,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        stage_depths: Union[int, List[int], Tuple[int, ...], None] = None,
        num_heads: Union[int, List[int], Tuple[int, ...], None] = None,
        window_size: Union[int, List[int], Tuple[int, ...]] = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]] = 2,
    ):
        super().__init__()

        self.key_to_encoder = "encoder"
        self.key_to_stem = "encoder.stem"
        self.keys_to_in_proj = ("encoder.stem",)
        self.key_to_lpe = None
        self.deep_supervision = deep_supervision

        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if len(n_conv_per_stage_decoder) != (n_stages - 1):
            raise ValueError(
                "n_conv_per_stage_decoder must have n_stages - 1 entries "
                f"({n_stages - 1}), got {len(n_conv_per_stage_decoder)}"
            )

        self.encoder = LiteSwinEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            stage_depths=stage_depths,
            num_heads=num_heads,
            window_size=window_size,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            encoder_lite_modules_per_stage=[0] * n_stages,
            lite_expansion_ratio=1.0,
            lite_se_reduction=4,
        )

        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        if len(input_size) != convert_conv_op_to_dim(self.encoder.conv_op):
            raise AssertionError(
                "just give the image size without color/feature channels or batch channel. "
                "Give input_size=(x, y(, z))"
            )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    def compute_approx_flops(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        if len(input_size) != convert_conv_op_to_dim(self.encoder.conv_op):
            raise AssertionError(
                "just give the image size without color/feature channels or batch channel. "
                "Give input_size=(x, y(, z))"
            )
        decoder_flops = 0
        if hasattr(self.decoder, "compute_approx_flops"):
            decoder_flops = int(self.decoder.compute_approx_flops(input_size))
        return int(self.encoder.compute_approx_flops(input_size) + decoder_flops)

    def estimate_computational_cost(self, input_size: Union[List[int], Tuple[int, ...]], batch_size: int = 1) -> dict:
        return estimate_cost_from_flops(self.compute_approx_flops(input_size), batch_size=batch_size)

    @staticmethod
    def initialize(module: nn.Module):
        if isinstance(
            module,
            (
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            ),
        ):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LayerNorm, LayerNormNd)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
