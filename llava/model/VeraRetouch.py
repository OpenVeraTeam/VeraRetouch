from .language_model.llava_qwen import *
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.generation.utils import GenerateOutput
from utils import get_model as get_retouch_decoder_model
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from transformers.utils import logging
from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.llava_arch import unpad_image
logger = logging.get_logger(__name__)
from utils import tensor2img
import cv2


class VeraRetouchForCausalLLM_Unified(LlavaQwen2ForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, config_add, step=None, writer=None):
        super().__init__(config)
        self.config_add = config_add
        self.retouch_decoder = get_retouch_decoder_model(config_add.retouch_decoder_name, logger).decoder
        self.ce_loss_weight = config_add.get('ce_loss_weight', None)
        self.token_loss_weight = config_add.get('token_loss_weight', None)
        self.image_loss_weight = config_add.get('image_loss_weight', None)
        self.tb_step = step
        self.writer = writer
        self.retouch_head = nn.Sequential(
            nn.Linear(config_add.retouch_head_in_dim, config_add.retouch_head_hidden_dim),
            nn.LayerNorm(config_add.retouch_head_hidden_dim),
            nn.GELU(),
            nn.Linear(config_add.retouch_head_hidden_dim, config_add.retouch_head_hidden_dim),
            nn.LayerNorm(config_add.retouch_head_hidden_dim),
            nn.GELU(),
            nn.Linear(config_add.retouch_head_hidden_dim, config_add.retouch_head_out_dim)
        )

    def register_special_token_idx(
        self,
        retouch_token_light_idx: int,
        retouch_token_colortemp_idx: int,
        retouch_token_colormixer_idx: int,
        task_auto_idx: int = None,
        task_style_idx: int = None,
        task_professional_idx: int = None,
        problem_light_start_idx: int = None,
        problem_globalcolor_start_idx: int = None,
        problem_specificcolor_start_idx: int = None,
        problem_light_end_idx: int = None,
        problem_globalcolor_end_idx: int = None,
        problem_specificcolor_end_idx: int = None,
        plan_light_start_idx: int = None,
        plan_globalcolor_start_idx: int = None,
        plan_specificcolor_start_idx: int = None,
        plan_light_end_idx: int = None,
        plan_globalcolor_end_idx: int = None,
        plan_specificcolor_end_idx: int = None,
    ):
        self.retouch_token_light_idx = retouch_token_light_idx
        self.retouch_token_colortemp_idx = retouch_token_colortemp_idx
        self.retouch_token_colormixer_idx = retouch_token_colormixer_idx

        self.task_auto_idx = task_auto_idx
        self.task_style_idx = task_style_idx
        self.task_professional_idx = task_professional_idx
        self.problem_light_start_idx = problem_light_start_idx
        self.problem_globalcolor_start_idx = problem_globalcolor_start_idx
        self.problem_specificcolor_start_idx = problem_specificcolor_start_idx
        self.problem_light_end_idx = problem_light_end_idx
        self.problem_globalcolor_end_idx = problem_globalcolor_end_idx
        self.problem_specificcolor_end_idx = problem_specificcolor_end_idx
        self.plan_light_start_idx = plan_light_start_idx
        self.plan_globalcolor_start_idx = plan_globalcolor_start_idx
        self.plan_specificcolor_start_idx = plan_specificcolor_start_idx
        self.plan_light_end_idx = plan_light_end_idx
        self.plan_globalcolor_end_idx = plan_globalcolor_end_idx
        self.plan_specificcolor_end_idx = plan_specificcolor_end_idx
    
    def freeze_retouch_decoder(self):
        for param in self.retouch_decoder.parameters():
            param.requires_grad = False
        print("Freeze Retouch Decoder!!!")
    
    def freeze_retouch_head(self):
        for param in self.retouch_head.parameters():
            param.requires_grad = False
        print("Freeze Retouch Head!!!")

    def unfreeze_retouch_decoder(self):
        for param in self.retouch_decoder.parameters():
            param.requires_grad = True
        print("Unfreeze Retouch Decoder!!!")
        
    def unfreeze_llm(self):
        for name, p in self.get_model().named_parameters():
            if "vision_tower" not in name:
                if "mm_projector" not in name:
                    p.requires_grad = True
        print("Unfreeze LLM!!!")
    
    def freeze_llm(self):
        for name, p in self.get_model().named_parameters():
            if "vision_tower" not in name:
                if "mm_projector" not in name:
                    p.requires_grad = False
        print("Freeze LLM!!!")
            
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, ref_retouch_tokens=None, image_sizes=None, batch_infer=False,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            if hasattr(self.get_vision_tower(), 's2_image_size'):
                                img_size = self.get_vision_tower().s2_image_size
                            elif isinstance(self.get_vision_tower().config, dict):
                                img_size = self.get_vision_tower().config["image_cfg"]["image_size"]
                            else:
                                img_size = self.get_vision_tower().config.image_size

                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, img_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            # Get the indices of image tokens in the sequence, add boundary values at the beginning and end for easy segmentation
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            
            # Split text segments without image tokens and corresponding labels
            cur_input_ids_noim = [] # Store split text tokens
            cur_labels = labels[batch_idx] # Original labels for current batch
            cur_labels_noim = [] # Store split text labels
            for i in range(len(image_token_indices) - 1):
                # Extract text tokens between two image tokens (e.g., [start, end) interval)
                # Example: If input sequence is [TextA, ImageToken1, TextB, ImageToken2, TextC], then image_token_indices is [-1, 1, 3, 5], and the split text segments are [TextA, TextB, TextC].
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_ids_noim = torch.cat(cur_input_ids_noim)
            cur_input_embeds = self.get_model().embed_tokens(cur_input_ids_noim)
            ### Replace <retouch_token> embedding with retouch tokens
            # retouch_token_mask = cur_input_ids_noim == self.retouch_token_occ_idx
            # cur_input_embeds[retouch_token_mask] = ref_retouch_tokens[batch_idx].view(-1, cur_input_embeds.shape[1])
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = [] # Store new embeddings (text + image) for current batch
            cur_new_labels = [] # Store new labels for current batch

            # Number of iterations = number of text segments (num_images + 1, since n image tokens split into n+1 text segments)
            for i in range(num_images + 1):
                # First add the embedding and labels of the i-th text segment
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                # Create labels for image tokens: fill with IGNORE_INDEX since image features don't need loss calculation
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        # Pad back to max length
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)


        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left" or batch_infer:
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        
        # if batch_infer:
        #     pass
        # elif _position_ids is None:
        #     position_ids = None
        if _position_ids is None:
            position_ids = None
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    

    @torch.no_grad()
    def _generate(
        self,
        tokenizer,
        inputs: Optional[torch.Tensor] = None,
        # images: Optional[torch.Tensor] = None,
        images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        image_sizes: Optional[torch.Tensor] = None,
        retouch_masks: torch.FloatTensor = None,
        input_imgs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        chunk: Optional[int] = -1,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # inputs__ = inputs
        # attention_mask__ = attention_mask
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                batch_infer=True,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        outputs =  Qwen2ForCausalLM.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        # Assume output_ids shape is [bs, seq_len], hidden_states is a tuple (each element shape is [bs, seq_len, hidden_size])
        output_ids = outputs.sequences  # [bs, seq_len]
        hidden_states = outputs.hidden_states  # Tuple where each element is a tensor with shape (batch_size, total sequence length up to current step, hidden_size)
        bs = output_ids.shape[0]  
        # 1. Generate batch masks (shape remains [bs, seq_len], no dimension squeezing)
        retouch_token_light_mask = output_ids == self.retouch_token_light_idx  # [bs, seq_len]
        retouch_token_colortemp_mask = output_ids == self.retouch_token_colortemp_idx  # [bs, seq_len]
        retouch_token_colormixer_mask = output_ids == self.retouch_token_colormixer_idx  # [bs, seq_len]

        # Store results for each sample
        retouched_imgs = []
        decoded_outputs = []

        # 2. Iterate over each sample in the batch
        for i in range(bs):
            # 2.1 Extract mask for current sample (take i-th sample, shape [seq_len])
            light_mask = retouch_token_light_mask[i]  # [seq_len]
            colortemp_mask = retouch_token_colortemp_mask[i]  # [seq_len]
            colormixer_mask = retouch_token_colormixer_mask[i]  # [seq_len]
            
            # 2.2 Get indices where conditions are met for current sample (take 0-th dimension indices, i.e., sequence positions)
            # Note: If a sample has no corresponding token, handle empty indices (assume each sample has at least one here)
            light_indices = torch.where(light_mask)[0]  # [N1], N1 is the number of light tokens in current sample
            colortemp_indices = torch.where(colortemp_mask)[0]  # [N2]
            colormixer_indices = torch.where(colormixer_mask)[0]  # [N3]
            
            # Get hidden state corresponding to the first token that meets the condition for each sample ([-1] represents the last layer hidden state)
            # Each element in hidden_states tuple is [bs, seq_len, hidden_size], take the i-th sample's hidden state
            light_latent = hidden_states[light_indices[0]][-1][i].squeeze()  # [hidden_size]
            colortemp_latent = hidden_states[colortemp_indices[0]][-1][i].squeeze()  # [hidden_size]
            colormixer_latent = hidden_states[colormixer_indices[0]][-1][i].squeeze()  # [hidden_size]
            
            # 2.3 Concatenate and process latent for current sample
            retouch_latent = torch.stack([light_latent, colortemp_latent, colormixer_latent]).view(-1).unsqueeze(0)  # [1, 3*hidden_size]
            retouch_latent = self.retouch_head(retouch_latent)  # Assume output shape [1, ...]
            if retouch_masks != None:
                mask_expend = retouch_masks[i].unsqueeze(-1).expand(-1, int(retouch_latent.shape[-1]/3)).reshape(retouch_latent.shape)
                retouch_latent = retouch_latent * mask_expend
            # 2.4 Generate retouched result for current sample (input_img should correspond to the i-th image in the batch, assume input_img shape is [bs, ...])
            input_img = input_imgs[i].unsqueeze(0)
            if chunk > 0:
                retouched_img = self.retouch_decoder.forward_chunk(input_img, retouch_latent, chunk).to(torch.float32)  # Keep batch dimension as 1
            else:
                retouched_img = self.retouch_decoder(input_img, retouch_latent).to(torch.float32)
            retouched_img = cv2.cvtColor(tensor2img(retouched_img), cv2.COLOR_RGB2BGR)
            retouched_imgs.append(retouched_img)
            
            # 2.5 Decode text output for current sample
            decoded = tokenizer.batch_decode(output_ids[i:i+1], skip_special_tokens=True)[0].strip()
            decoded_outputs.append(decoded)

        return retouched_imgs, decoded_outputs
    