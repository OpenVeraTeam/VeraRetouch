from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from .resnet import *
from .colormlp_v2 import *
from .vgg import *
from diffusers.utils import BaseOutput


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None

class SimplifiedUnet(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.Tensor,
        encoder_hidden_states: torch.Tensor,  # Should be (batch, seq_len, 896)
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # Remove upsample factor check (can keep if needed)
        
        # 0. Center input (if needed)
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. Use zero embedding instead of time embedding
        batch_size = sample.shape[0]
        device = sample.device
        dtype = sample.dtype
        
        # Create zero tensor matching original embedding dimension
        emb = torch.zeros(
            (batch_size, self.time_embedding.linear_1.out_features),
            device=device,
            dtype=dtype
        )
        
        # 2. Preprocessing
        sample = self.conv_in(sample)
        
        # 3. Downsampling
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,  # Pass zero embedding
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=None,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb  # Pass zero embedding
                )
            down_block_res_samples += res_samples

        # 4. Mid block
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,  # Pass zero embedding
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=None,
                )
            else:
                sample = self.mid_block(sample, emb)  # Pass zero embedding

        # 5. Upsampling
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,  # Pass zero embedding
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=None,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,  # Pass zero embedding
                    res_hidden_states_tuple=res_samples,
                )

        # 6. Postprocessing
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

        
        
# Create initialization function for tone transfer UNet
def create_tone_transfer_unet(
    in_channels: int = 3,
    out_channels: int = 3,
    block_out_channels: tuple = (64, 128, 256, 512),
    layers_per_block: int = 2,
    cross_attention_dim: int = 896,
    attention_head_dim: int = 8,
    norm_num_groups: int = 32,
) -> SimplifiedUnet:
    """
    Create a simplified UNet for tone transfer
    
    Args:
        in_channels: Input image channels (default: 3)
        out_channels: Output image channels (default: 3)
        block_out_channels: Output channels for each downsampling block (default: (64, 128, 256, 512))
        layers_per_block: Number of residual layers per block (default: 2)
        cross_attention_dim: Cross attention feature dimension (default: 896)
        attention_head_dim: Attention head dimension (default: 8)
        norm_num_groups: Number of normalization groups (default: 32)
    
    Returns:
        Configured tone transfer UNet model
    """
    # Create config object
    config = {
        "sample_size": 256,  # Can be adjusted based on actual image size
        "in_channels": in_channels,
        "out_channels": out_channels,
        "center_input_sample": False,  # Disable input centering
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "down_block_types": (
            "DownBlock2D",  # First layer without cross attention
            "CrossAttnDownBlock2D",  # Subsequent layers with cross attention
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        "up_block_types": (
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",  # Last layer without cross attention
        ),
        "block_out_channels": block_out_channels,
        "layers_per_block": layers_per_block,
        "downsample_padding": 1,
        "mid_block_scale_factor": 1,
        "act_fn": "silu",
        "norm_num_groups": norm_num_groups,
        "norm_eps": 1e-5,
        "cross_attention_dim": cross_attention_dim,
        "attention_head_dim": attention_head_dim,
        "dual_cross_attention": False,
        "use_linear_projection": False,
        "class_embed_type": None,  # Disable class embedding
        "addition_embed_type": None,  # Disable additional embedding
        "num_class_embeds": None,
        "upcast_attention": False,
        "resnet_time_scale_shift": "default",
    }
    
    # Create and return UNet instance
    return SimplifiedUnet(**config)


class ColorIlluminationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor (weight shared)
        self.feature_extractor = nn.Sequential(
            # Stage 1: 256x256
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Stage 2: 64x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 3: 32x32 (increase depth to preserve information)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 4: 16x16 (final feature size)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion and encoder
        self.diff_encoder = nn.Sequential(
            # Feature fusion: [original1, original2, absolute diff, relative diff]
            nn.Conv2d(1024, 512, kernel_size=1),  # 1x1 conv for dimension reduction
            
            # Spatial information aggregation
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Multi-scale feature extraction
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            
            # Final encoding layer
            nn.Conv2d(512, 896, kernel_size=3, padding=1),
            nn.BatchNorm2d(896),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, img_orj, img_adj):
        # Extract features
        feat1 = self.feature_extractor(img_orj)  # [B,256,16,16]
        feat2 = self.feature_extractor(img_adj)  # [B,256,16,16]
        
        # Compute multiple difference features
        diff = feat2 - feat1
        rel_diff = feat2 / (feat1 + 1e-6)  # Relative difference
        
        # Quadruple feature fusion
        combined = torch.cat([feat1, feat2, diff, rel_diff], dim=1)  # 1024 channels
        
        # Encode to 896-dimensional token
        return self.diff_encoder(combined).unsqueeze(1)

class RetouchRenderer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cat3=False
        encoder_args = args.encoder_args
        decoder_args = args.decoder_args
        if encoder_args.model_name == "SiglipEncoder":
            self.encoder = SiglipEncoder(encoder_args.latent_dim, encoder_args.num_tokens, encoder_args.lora_config, encoder_args.concat_dim, encoder_args.local_files_only)
        elif encoder_args.model_name == "SiglipEncoderDual":
            self.encoder = SiglipEncoderDual(encoder_args.latent_dim, encoder_args.num_tokens, encoder_args.lora_config, encoder_args.concat_dim, encoder_args.local_files_only)
        elif encoder_args.model_name == "VGG16Encoder":
            self.encoder = VGG16Encoder(encoder_args.latent_dim)
        elif encoder_args.model_name == "VGG19Encoder":
            self.encoder = VGG19Encoder(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder":
            self.encoder = Resnet18Encoder(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18_InteractEncoder":
            self.encoder = Resnet18_InteractEncoder(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18_CrossInteractEncoder":
            self.encoder = Resnet18_CrossInteractEncoder(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet22Encoder":
            self.encoder = Resnet22Encoder(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_CBAM":
            self.encoder = Resnet18Encoder_CBAM(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_MixedCBAM":
            self.encoder = Resnet18Encoder_MixedCBAM(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_InputRes":
            self.encoder = Resnet18Encoder_InputRes(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_SpecialAtten":
            self.encoder = Resnet18Encoder_SpecialAtten(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_InputCatMixedCBAM":
            self.encoder = Resnet18Encoder_InputCatMixedCBAM(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_3InputCatMixedCBAM":
            self.cat3=True
            self.encoder = Resnet18Encoder_3InputCatMixedCBAM(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_InputCatMixedCBAMFPN":
            self.encoder = Resnet18Encoder_InputCatMixedCBAMFPN(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_FPN":
            self.encoder = Resnet18Encoder_FPN(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_MinusMixedCBAM":
            self.encoder = Resnet18Encoder_MinusMixedCBAM(encoder_args.latent_dim)
        elif encoder_args.model_name == "Resnet18Encoder_2NetMinusMixedCBAM":
            self.encoder = Resnet18Encoder_2NetMinusMixedCBAM(encoder_args.latent_dim)
        if decoder_args.name == "ConditionalMLPDecoderDualAdaLN":
            self.decoder = ConditionalMLPDecoderDualAdaLN(decoder_args.latent_dim, decoder_args.hidden_dims, decoder_args.cond_method, decoder_args.final_act)
        else:     
            self.decoder = ConditionalMLPDecoder(decoder_args.latent_dim, decoder_args.hidden_dims, decoder_args.cond_method, decoder_args.final_act)

    def forward(self, x, ref_o, ref_t, mask, chunk=-1):
        if self.cat3:
            control_feat = self.encoder(ref_o, ref_t, x)
        else:
            control_feat = self.encoder(ref_o, ref_t)
        mask_expend = mask.unsqueeze(-1).expand(-1, -1, self.decoder.latent_dim).reshape(mask.shape[0], -1)
        if len(control_feat.shape) == 3: # Dual AdaLN
            mask_expend = mask_expend.unsqueeze(1).expand(-1, 2, -1)
        control_feat = control_feat * mask_expend
        if chunk > 0:
            output = self.decoder.forward_chunk(x, control_feat, chunk)
        else:
            output = self.decoder(x, control_feat)
        return output
    

class ParamPredWrap(nn.Module):
    def __init__(self, retouch_renderer, num_params, latent_dim):
        super().__init__()
        self.retouch_renderer = retouch_renderer
        self.num_params = num_params
        self.latent_dim = latent_dim
        self.map_layers = Sequential(
            Linear(latent_dim * 3, latent_dim),
            ReLU(),
            Linear(latent_dim, num_params),
        )
    
    def forward(self, x, ref_o, ref_t, mask):
        control_feat = self.retouch_renderer.encoder(ref_o, ref_t)
        mask_expend = mask.unsqueeze(-1).expand(-1, -1, self.latent_dim).reshape(mask.shape[0], -1)
        if len(control_feat.shape) == 3: # Dual AdaLN
            mask_expend = mask_expend.unsqueeze(1).expand(-1, 2, -1)
        control_feat = control_feat * mask_expend
        param_pred = torch.sigmoid(self.map_layers(control_feat))
        output = self.retouch_renderer.decoder(x, control_feat)
        return output, param_pred


    



