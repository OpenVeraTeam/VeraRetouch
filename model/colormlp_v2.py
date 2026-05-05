
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoProcessor, SiglipVisionModel 
from peft import LoraConfig, get_peft_model
from .model_utils import tensor2img
from .resnet import Resnet18Encoder, Resnet18_InteractEncoder
from .vgg import *


class SiglipEncoder(nn.Module):

    def __init__(self, latent_dim=1024, num_tokens=8, lora_config=None, concat_dim=1, local_files_only=False):

        super().__init__()
        self.backbone = SiglipVisionModel.from_pretrained("/data/vjuicefs_ai_camera_llm/intern_data/yihong/models/siglip2-so400m-patch14-384", local_files_only=local_files_only)
        for param in self.backbone.parameters():
                param.requires_grad = False
        if lora_config is not None:
            lora_config = LoraConfig(**lora_config)
            self.backbone = get_peft_model(self.backbone, lora_config)
        
        self.img_processor = AutoProcessor.from_pretrained("/data/vjuicefs_ai_camera_llm/intern_data/yihong/models/siglip2-so400m-patch14-384", local_files_only=local_files_only)
        self.num_tokens = num_tokens
        self.concat_dim = concat_dim

        if self.num_tokens is not None:
            self.proj = nn.Linear(1152*2 if concat_dim==2 else 1152, latent_dim*2)
            self.latents = nn.Parameter(torch.randn(num_tokens, latent_dim*2))
            self.perceiver_sampler = nn.MultiheadAttention(
                embed_dim=latent_dim*3, 
                num_heads=4, 
                batch_first=True
            )
        else:
            self.proj = nn.Linear(1152*2, latent_dim*3)
            if self.concat_dim != 1:
                print("Warning: concat_dim is not 1, but num_tokens is None. Setting concat_dim to 1.")
            self.concat_dim = 1 

    def forward(self, x, y):
        device = x.device 
        x = tensor2img(x, input_scale=(-1, 1))
        y = tensor2img(y, input_scale=(-1, 1))
        x = self.img_processor(images=x, return_tensors="pt").pixel_values
        y = self.img_processor(images=y, return_tensors="pt").pixel_values

        
        if self.num_tokens is not None:
            x = self.backbone(x.to(device)).last_hidden_state
            y = self.backbone(y.to(device)).last_hidden_state
            x = torch.cat([x, y], dim=self.concat_dim)
            x = self.proj(x)
            x = self.perceiver_sampler(self.latents.expand(x.shape[0], -1, -1).to(device), x, x, need_weights=False)[0]
        else:
            x = self.backbone(x.to(device)).pooler_output
            y = self.backbone(y.to(device)).pooler_output
            x = torch.cat([x, y], dim=self.concat_dim)
            x = self.proj(x)

        return x 

class SiglipEncoder(nn.Module):

    def __init__(self, latent_dim=1024, num_tokens=8, lora_config=None, concat_dim=1, local_files_only=False):

        super().__init__()
        self.backbone = SiglipVisionModel.from_pretrained("/data/vjuicefs_ai_camera_llm/intern_data/yihong/models/siglip2-so400m-patch14-384", local_files_only=local_files_only)
        for param in self.backbone.parameters():
                param.requires_grad = False
        if lora_config is not None:
            lora_config = LoraConfig(**lora_config)
            self.backbone = get_peft_model(self.backbone, lora_config)
        
        self.img_processor = AutoProcessor.from_pretrained("/data/vjuicefs_ai_camera_llm/intern_data/yihong/models/siglip2-so400m-patch14-384", local_files_only=local_files_only)
        self.num_tokens = num_tokens
        self.concat_dim = concat_dim

        if self.num_tokens is not None:
            self.proj = nn.Linear(1152*2 if concat_dim==2 else 1152, latent_dim*2)
            self.latents = nn.Parameter(torch.randn(num_tokens, latent_dim*2))
            self.perceiver_sampler = nn.MultiheadAttention(
                embed_dim=latent_dim*3, 
                num_heads=4, 
                batch_first=True
            )
        else:
            self.proj = nn.Linear(1152*2, latent_dim*3)
            if self.concat_dim != 1:
                print("Warning: concat_dim is not 1, but num_tokens is None. Setting concat_dim to 1.")
            self.concat_dim = 1 

    def forward(self, x, y):
        device = x.device 
        x = tensor2img(x, input_scale=(-1, 1))
        y = tensor2img(y, input_scale=(-1, 1))
        x = self.img_processor(images=x, return_tensors="pt").pixel_values
        y = self.img_processor(images=y, return_tensors="pt").pixel_values
        
        if self.num_tokens is not None:
            x = self.backbone(x.to(device)).last_hidden_state
            y = self.backbone(y.to(device)).last_hidden_state
            x = torch.cat([x, y], dim=self.concat_dim)
            x = self.proj(x)
            x = self.perceiver_sampler(self.latents.expand(x.shape[0], -1, -1).to(device), x, x, need_weights=False)[0]
        else:
            x = self.backbone(x.to(device)).pooler_output
            y = self.backbone(y.to(device)).pooler_output
            x = torch.cat([x, y], dim=self.concat_dim)
            x = self.proj(x)

        return x 

class SiglipEncoderDual(nn.Module):

    def __init__(self, latent_dim=1024, num_tokens=8, lora_config=None, concat_dim=1, local_files_only=False):

        super().__init__()
        self.backbone = SiglipVisionModel.from_pretrained("/data/vjuicefs_ai_camera_llm/intern_data/yihong/models/siglip2-so400m-patch14-384", local_files_only=local_files_only)
        for param in self.backbone.parameters():
                param.requires_grad = False
        if lora_config is not None:
            lora_config = LoraConfig(**lora_config)
            self.backbone = get_peft_model(self.backbone, lora_config)
        
        self.img_processor = AutoProcessor.from_pretrained("/data/vjuicefs_ai_camera_llm/intern_data/yihong/models/siglip2-so400m-patch14-384", local_files_only=local_files_only)
        self.num_tokens = num_tokens
        self.concat_dim = concat_dim

        if self.num_tokens is not None:
            self.proj = nn.Linear(1152*2 if concat_dim==2 else 1152, latent_dim*2)
            self.latents = nn.Parameter(torch.randn(num_tokens, latent_dim*2))
            self.perceiver_sampler = nn.MultiheadAttention(
                embed_dim=latent_dim*3, 
                num_heads=4, 
                batch_first=True
            )
        else:
            self.proj1 = nn.Linear(1152, latent_dim*3)
            self.proj2 = nn.Linear(1152, latent_dim*3)
            if self.concat_dim != 1:
                print("Warning: concat_dim is not 1, but num_tokens is None. Setting concat_dim to 1.")
            self.concat_dim = 1 

    def forward(self, x, y):
        device = x.device 
        x = tensor2img(x, input_scale=(-1, 1))
        y = tensor2img(y, input_scale=(-1, 1))
        x = self.img_processor(images=x, return_tensors="pt").pixel_values
        y = self.img_processor(images=y, return_tensors="pt").pixel_values

        
        x = self.backbone(x.to(device)).pooler_output
        y = self.backbone(y.to(device)).pooler_output
        x = self.proj1(x).unsqueeze(1)
        y = self.proj2(y).unsqueeze(1)
        x = torch.cat([x, y], dim=self.concat_dim)
        return x 

class ConditionalMLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        hidden_dims=[256, 512, 1024],
        cond_method='add',
        final_act='sigmoid'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_method = cond_method

        in_dim = 3
        self.layers = nn.ModuleList()
        for k, h in enumerate(hidden_dims):
            self.layers.append(nn.Linear(in_dim, h))
            if cond_method=='cat':
                in_dim = h * 2
            else:
                in_dim = h

    
        self.out_layer = nn.Linear(in_dim, 3)
        
        self.z_projs = nn.ModuleList()
        for h in hidden_dims:
            if cond_method == "cross_attn":
                self.z_projs.append(CrossAttention(
                    query_dim=h,
                    condition_dim=latent_dim,
                    out_dim=h
                ))
            else:
                hid_dim = h * 2 if cond_method == 'adain' else h
                self.z_projs.append(nn.Linear(latent_dim * 3, hid_dim))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(h) for h in hidden_dims])
        self.final_act = final_act

        
   
    def forward(self, x, z):

        x_shape = x.shape 
        x = x.view(x_shape[0], x_shape[1], -1).permute(0, 2, 1)

        for z_proj, layer_norm, layer in zip(self.z_projs, self.layer_norms, self.layers):
            x = F.relu(layer(x))
            x = layer_norm(x)

            if self.cond_method == 'add':
                x = x + z_proj(F.relu(z)).unsqueeze(1)
            elif self.cond_method == "cross_attn":
                x = x + z_proj(x, z)
            elif self.cond_method == "cat":
                k = z_proj(F.relu(z)).unsqueeze(1)
                k = k.expand(-1, x.shape[1], -1)  # 复制到 [bs, 262144, hid_dim]
                x = torch.cat([x, k], dim=-1)
            else:
                z_scale, z_shift = z_proj(F.relu(z)).chunk(2, dim=-1) # I follow flux for this 
                x = x * (1 + z_scale.unsqueeze(1)) + z_shift.unsqueeze(1) 

        x = self.out_layer(x)
        if self.final_act.lower() == "none":
            pass
        elif self.final_act.lower() == "sigmoid":
            x = torch.sigmoid(x)
        elif self.final_act.lower() == "hard_sigmoid":
            x = F.hardsigmoid(x)
        elif self.final_act.lower() == "hard_tanh":
            x = F.hardtanh(x)*1.1
            x = (x + 1.0) * 0.5

        x = x.permute(0, 2, 1).view(x_shape)
        
        return x
    
    def forward_chunk(self, x, z, chunk_size=65536):  # 新增 chunk_size：每个块的空间维度大小，可按需调整
        x_shape = x.shape  # 保存初始形状：(bs, C, H, W)
        bs, C, H, W = x_shape
        spatial_dim = H * W  # 总空间像素数

        # 步骤1：展平+维度置换（与原逻辑一致）
        x = x.view(bs, C, -1).permute(0, 2, 1)  # (bs, H*W, C)

        # 步骤2：初始化输出张量（提前分配内存，避免拼接时显存碎片）
        x_output = torch.zeros_like(x)  # (bs, H*W, C)，与分块计算前的x形状一致

        # 步骤3：分块遍历计算（核心：按空间维度切分，逐块处理）
        for start_idx in range(0, spatial_dim, chunk_size):
            # 计算当前块的结束索引（防止最后一个块超出边界）
            end_idx = min(start_idx + chunk_size, spatial_dim)
            # 切分当前块：仅在空间维度（第1维）上分块
            x_chunk = x[:, start_idx:end_idx, :]  # (bs, chunk_len, C)，chunk_len ≤ chunk_size

            # 步骤4：对当前块执行原有的MLP+条件融合逻辑（完全复用原逻辑，仅输入为小块）
            for z_proj, layer_norm, layer in zip(self.z_projs, self.layer_norms, self.layers):
                x_chunk = F.relu(layer(x_chunk))
                x_chunk = layer_norm(x_chunk)

                if self.cond_method == 'add':
                    # z特征广播适配当前块形状（自动兼容chunk_len）
                    z_feat = z_proj(F.relu(z)).unsqueeze(1)  # (bs, 1, hid_dim)
                    x_chunk = x_chunk + z_feat
                elif self.cond_method == "cross_attn":
                    # 交叉注意力仅作用于当前块，显存占用大幅降低
                    x_chunk = x_chunk + z_proj(x_chunk, z)
                elif self.cond_method == "cat":
                    z_feat = z_proj(F.relu(z)).unsqueeze(1)  # (bs, 1, hid_dim)
                    # 仅扩展到当前块的空间长度（chunk_len），而非全部 H*W
                    z_feat = z_feat.expand(-1, x_chunk.shape[1], -1)  # (bs, chunk_len, hid_dim)
                    x_chunk = torch.cat([x_chunk, z_feat], dim=-1)  # (bs, chunk_len, hid_dim*2)
                else:
                    # 缩放平移模式，同样适配当前块
                    z_scale, z_shift = z_proj(F.relu(z)).chunk(2, dim=-1)  # (bs, hid_dim) each
                    z_scale = z_scale.unsqueeze(1)  # (bs, 1, hid_dim)
                    z_shift = z_shift.unsqueeze(1)  # (bs, 1, hid_dim)
                    x_chunk = x_chunk * (1 + z_scale) + z_shift

            # 步骤5：当前块通过输出层，还原通道维度
            x_chunk = self.out_layer(x_chunk)  # (bs, chunk_len, C)

            # 步骤6：激活函数（复用原逻辑，不改变形状）
            if self.final_act.lower() == "sigmoid":
                x_chunk = torch.sigmoid(x_chunk)
            elif self.final_act.lower() == "hard_sigmoid":
                x_chunk = F.hardsigmoid(x_chunk)
            elif self.final_act.lower() == "hard_tanh":
                x_chunk = F.hardtanh(x_chunk) * 1.1
                x_chunk = (x_chunk + 1.0) * 0.5
            # "none" 模式无需处理

            # 步骤7：将当前块结果写入输出张量对应位置
            x_output[:, start_idx:end_idx, :] = x_chunk

        # 步骤8：用分块计算后的完整输出还原图像形状（与原逻辑一致）
        x = x_output.permute(0, 2, 1).view(x_shape)  # (bs, C, H, W)

        return x

class ConditionalMLPDecoder_Conv2D(nn.Module):
    def __init__(
        self,
        latent_dim=896,
        hidden_dims=[256, 512, 1024],
        cond_method='add',
        final_act='sigmoid'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_method = cond_method

        in_channels = 3
        self.layers = nn.ModuleList()
        for k, h in enumerate(hidden_dims):
            self.layers.append(nn.Conv2d(in_channels, h, kernel_size=1))
            if cond_method=='cat':
                in_channels = h * 2
            else:
                in_channels = h

    
        self.out_layer = nn.Conv2d(in_channels, 3, kernel_size=1)
        
        self.z_projs = nn.ModuleList()
        for h in hidden_dims:
            if cond_method == "add":
                hid_dim = h * 2 if cond_method == 'adain' else h
                self.z_projs.append(nn.Linear(latent_dim * 3, hid_dim))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(h) for h in hidden_dims])
        self.final_act = final_act

        
   
    def forward(self, x, z):

        x_shape = x.shape 

        for z_proj, layer_norm, layer in zip(self.z_projs, self.layer_norms, self.layers):
            x = F.relu(layer(x))
            # 调整形状以应用LayerNorm (batch, height*width, h)
            b, c, h, w = x.shape
            x_reshaped = x.view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, c)
            x_reshaped = layer_norm(x_reshaped)
            x = x_reshaped.permute(0, 2, 1).view(b, c, h, w)  # 恢复为4D

            if self.cond_method == 'add':
                x = x + z_proj(F.relu(z)).unsqueeze(-1).unsqueeze(-1)
            elif self.cond_method == "cat":
                z_feat = z_proj(F.relu(z)).unsqueeze(-1).unsqueeze(-1)  # (batch, h, 1, 1)
                z_feat = z_feat.expand(-1, -1, x.shape[2], x.shape[3])  # 扩展到与x相同的空间维度
                x = torch.cat([x, z_feat], dim=1)  # 在通道维度拼接

        x = self.out_layer(x)
        if self.final_act.lower() == "none":
            pass
        elif self.final_act.lower() == "sigmoid":
            x = torch.sigmoid(x)
        elif self.final_act.lower() == "hard_sigmoid":
            x = F.hardsigmoid(x)
        elif self.final_act.lower() == "hard_tanh":
            x = F.hardtanh(x)*1.1
            x = (x + 1.0) * 0.5

        # 确保输出形状与输入一致
        assert x.shape == x_shape, f"输出形状 {x.shape} 与输入形状 {x_shape} 不匹配"
        return x

# if __name__ == "__main__":
#     model = ConditionalMLPDecoder_Conv2D()
#     input = torch.rand((8, 3, 512, 512))
#     z = torch.rand(8, 896*3)
#     output = model(input, z)
#     print(output.shape)
    
class ConditionalMLPDecoderDualAdaLN(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        hidden_dims=[256, 512, 1024],
        cond_method='add',
        final_act='sigmoid'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_method = cond_method

        in_dim = 3
        self.layers = nn.ModuleList()
        for k, h in enumerate(hidden_dims):
            self.layers.append(nn.Linear(in_dim, h))
            if cond_method=='cat':
                in_dim = h * 2
            else:
                in_dim = h

    
        self.out_layer = nn.Linear(in_dim, 3)
        
        self.z_projs_a = nn.ModuleList()
        self.z_projs_b = nn.ModuleList()
        for h in hidden_dims:
            hid_dim = h * 2
            self.z_projs_a.append(nn.Linear(latent_dim * 3, hid_dim))
            self.z_projs_b.append(nn.Linear(latent_dim * 3, hid_dim))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(h) for h in hidden_dims])
        self.final_act = final_act
        
   
    def forward(self, x, z):

        x_shape = x.shape 
        x = x.view(x_shape[0], x_shape[1], -1).permute(0, 2, 1)
        print(z.shape)
        z_a = z[:,0,:]
        z_b = z[:,1,:]

        for z_proj_a, z_proj_b, layer_norm, layer in zip(self.z_projs_a, self.z_projs_b, self.layer_norms, self.layers):
            x = F.relu(layer(x))
            x = layer_norm(x)

            z_scale_a_, z_shift_a = z_proj_a(F.relu(z_a)).chunk(2, dim=-1) # I follow flux for this
            z_scale_b, z_shift_b = z_proj_b(F.relu(z_b)).chunk(2, dim=-1) # I follow flux for this 
            x = (x - z_shift_a.unsqueeze(1)) * z_scale_a_.unsqueeze(1)  # 除法和乘法等价，为了防止/0采用乘法
            x = x * (1 + z_scale_b.unsqueeze(1)) + z_shift_b.unsqueeze(1) 

        x = self.out_layer(x)
        if self.final_act.lower() == "none":
            pass
        elif self.final_act.lower() == "sigmoid":
            x = torch.sigmoid(x)
        elif self.final_act.lower() == "hard_sigmoid":
            x = F.hardsigmoid(x)
        elif self.final_act.lower() == "hard_tanh":
            x = F.hardtanh(x)*1.1
            x = (x + 1.0) * 0.5

        x = x.permute(0, 2, 1).view(x_shape)

        return x
    

class CrossAttention(nn.Module):

    def __init__(self, query_dim, condition_dim, out_dim, num_heads=4):
        super().__init__()

        inner_dim = out_dim 
        head_dim = inner_dim // num_heads

        self.condition_norm = nn.LayerNorm(condition_dim)

        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(condition_dim, inner_dim)
        self.to_v = nn.Linear(condition_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.num_heads = num_heads
        self.head_dim = head_dim 


    def forward(self, x, condition):

        condition = self.condition_norm(condition)
        key = self.to_k(condition)
        value = self.to_v(condition)
        query = self.to_q(x) 

        query = query.view(x.shape[0], x.shape[1], 1, self.num_heads, self.head_dim).transpose(2, 3)
        key = key.view(condition.shape[0], 1, -1, self.num_heads, self.head_dim).transpose(2, 3)
        value = value.view(condition.shape[0], 1, -1, self.num_heads, self.head_dim).transpose(2, 3)

        output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        output = output.transpose(2, 3).reshape(x.shape[0], x.shape[1], self.num_heads * self.head_dim)
        output = self.to_out(output) 

        return output 





# class AsymmetricVAE(nn.Module):
    
#     def __init__(self, encoder_config, decoder_config):

#         super().__init__()

#         self.encoder = instantiate_from_config(encoder_config)
#         self.decoder = instantiate_from_config(decoder_config) 
#         self.reg = DiagonalGaussian()


#     def encode(self, x, y, sample=True):
#         z = self.encoder(x, y)
#         z = self.reg(z, sample=sample)
#         return z

#     def decode(self, x, z):
#         y_recon = self.decoder(x, z)
#         return y_recon 
    
#     def forward(self, x, y, sample=True):
#         return self.decode(x, self.encode(x, y, sample))

# if __name__ == "__main__": 


#     from PIL import Image
#     import numpy as np 

#     encoder_config = dict(
#         target="style_encoder.model.ResNetEncoder",
#         params=dict(
#             latent_dim=256,
#             pretrained=True,
#             num_tokens=None
#         )
#     )
#     encoder2_config = dict(
#         target="style_encoder.model.SiglipEncoder",
#         params=dict(
#             latent_dim=2048,
#             num_tokens=8,
#             lora_config=None,
#             concat_dim=1
#         )
#     )
#     decoder_cofing = dict(
#         target="style_encoder.model.ConditionalMLPDecoder",
#         params=dict(
#             latent_dim=2048,
#             hidden_dims=[128, 256, 512],
#             cond_method='ada_ln'
#         )
#     )
#     vae = AsymmetricVAE(encoder2_config, decoder_cofing)

#     x = torch.randn(8, 3, 256, 256)
#     y = torch.randn(8, 3, 256, 256)
#     y_recon = vae(x, y)
#     print(vae.reg.kl(), y_recon.shape)
#     print(sum([p.numel() for p in vae.parameters() if p.requires_grad]))


class ColorMLP_V2(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        hidden_dims = args.hidden_dims

        in_dim = 3
        self.layers = nn.ModuleList()
        for k, h in enumerate(hidden_dims):
            self.layers.append(nn.Linear(in_dim, h))
            in_dim = h 
    
        self.out_layer = nn.Linear(in_dim, 3)
        

        self.layer_norms = nn.ModuleList([nn.LayerNorm(h) for h in hidden_dims])

        
   
    def forward(self, x):

        x_shape = x.shape 
        x = x.view(x_shape[0], x_shape[1], -1).permute(0, 2, 1)

        for layer_norm, layer in zip(self.layer_norms, self.layers):
            x = F.relu(layer(x))
            x = layer_norm(x)

        x = self.out_layer(x)
        x = torch.sigmoid(x)

        x = x.permute(0, 2, 1).view(x_shape)

        return x


class Token_Optimize_Network(nn.Module):
    def __init__(
        self,
        z_dim,
        mlp_model,
        bs=1,
    ):
        super().__init__()  # 调用父类构造函数
        self.mlp = mlp_model
        self.bs = bs
        self.z_dim = z_dim
        
        # 冻结mlp_model的参数，不参与梯度更新
        for param in self.mlp.parameters():
            param.requires_grad = False
        
        # 定义可学习的z，形状为[bs, z_dim]
        self.z = nn.Parameter(torch.randn(bs, z_dim))  # 初始化为标准正态分布

    def init_z(self, mean=0.0, std=1.0):
        """用 torch.randn 的方式初始化 self.z"""
        with torch.no_grad():
            self.z.normal_(mean, std)   # 原地赋值，保持仍为 nn.Parameter

    def load_z(self, z, strict=True):
        if strict and z.shape != self.z.shape:
            raise RuntimeError(
                f"Shape mismatch: expected {self.z.shape}, got {z.shape}"
            )
        with torch.no_grad():
            self.z.copy_(z) 
            
    def forward(self, x):
        # 前向传播中使用冻结的mlp处理输入x
        x = self.mlp(x, self.z)
        
        return x  # 返回mlp的输出和可学习的z