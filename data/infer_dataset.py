import os
import cv2
import torch
from typing import Dict, Optional, Sequence, List
from PIL import Image
from torchvision import transforms
import json
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from llava.mm_utils import tokenizer_image_token, process_images_
from llava.conversation import conv_templates
from llava.constants import (DEFAULT_IMAGE_TOKEN,TASK_AUTO_RETOUCH_TOKEN, TASK_STYLE_RETOUCH_TOKEN,
                    TASK_PROFESSIONAL_RETOUCH_TOKEN)
from llava.mm_utils import tokenizer_image_token

class Infer_Auto_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths=[],
        model_path="./checkpoints/llava-fastvithd_0.5b_stage2",
        tokenizer=None,
        image_processor=None,
        samples_per_epoch=None,
        precision: str = "fp32",
        conv_mode: str = "qwen_2",
    ):
        super().__init__()
        self.conv_mode = conv_mode
        self.img_paths = img_paths
        
        self.image_processor = image_processor
        self.precision = precision
        if tokenizer and image_processor:
            self.tokenizer = tokenizer
            self.image_processor = image_processor
        else:
            print("Lack of tokenizer or image processor.")

    def resize2_512p(self, image):
        target_short_edge = 512
        original_width, original_height = image.size
        w, h = original_width, original_height
        if w <= h:
            scale = target_short_edge / w
            target_w = target_short_edge
            target_h = int(h * scale)
        else:
            scale = target_short_edge / h
            target_h = target_short_edge
            target_w = int(w * scale)
        image_resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return image_resized
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        qs = f"{TASK_AUTO_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. When I provide an image, please state the problems found in the image (from 3 aspects: lighting, global_color, specific color), and give the solution and retouch tokens."
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[self.conv_mode].copy() #
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        input_img_path = self.img_paths[idx]
        input_img_ = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        input_img_tensor = transforms.ToTensor()(cv2.cvtColor(input_img_, cv2.COLOR_BGR2RGB))
        input_img = (input_img_tensor - 0.5) * 2
        # Load and preprocess image
        image = Image.open(input_img_path).convert('RGB')
        image_512p = self.resize2_512p(image)
        image_tensor = process_images_([image_512p], self.image_processor)[0]

        image_size = image_512p.size
        mask = torch.tensor([1.0, 1.0, 1.0]) # 1 means using
        file_name = os.path.basename(input_img_path).split('.')[0]
        return dict(input_ids=input_ids.squeeze(),
                    image=image_tensor,
                    image_size=image_size,
                    file_name=file_name,
                    input_img = input_img,
                    retouch_mask = mask,
                    input_img_path=input_img_path,
                    )

class Infer_Style_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths=[],
        prompts=[],
        model_path="./checkpoints/llava-fastvithd_0.5b_stage2",
        tokenizer=None,
        image_processor=None,
        samples_per_epoch=None,
        precision: str = "fp32",
        conv_mode: str = "qwen_2",
    ):
        super().__init__()
        self.conv_mode = conv_mode
        self.img_paths = img_paths
        self.prompts = prompts
        
        self.image_processor = image_processor
        self.precision = precision
        if tokenizer and image_processor:
            self.tokenizer = tokenizer
            self.image_processor = image_processor
        else:
            print("Lack of tokenizer or image processor.")

    def resize2_512p(self, image):
        target_short_edge = 512
        original_width, original_height = image.size
        w, h = original_width, original_height
        if w <= h:
            # 宽是短边，先将宽缩放到512，高按比例缩放
            scale = target_short_edge / w
            target_w = target_short_edge
            target_h = int(h * scale)
        else:
            # 高是短边，先将高缩放到512，宽按比例缩放
            scale = target_short_edge / h
            target_h = target_short_edge
            target_w = int(w * scale)
        # 若你的PIL版本较低，可使用Image.ANTIALIAS
        image_resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return image_resized
    
    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        instruct_content = self.prompts[idx]
        # qs = f"{TASK_STYLE_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. I will provide an image and an instruction, please give me a retouch plan and retouch tokens.\n Instruction: I want a '{style}' for this photo. {instruct_content}"
        qs = f"{TASK_STYLE_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. I will provide an image and an instruction, please give me a retouch plan and retouch tokens.\n Instruction: {instruct_content}"
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[self.conv_mode].copy() #
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        input_img_path = self.img_paths[idx]
        input_img_ = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        input_img_tensor = transforms.ToTensor()(cv2.cvtColor(input_img_, cv2.COLOR_BGR2RGB))
        input_img = (input_img_tensor - 0.5) * 2
        # Load and preprocess image
        image = Image.open(input_img_path).convert('RGB')
        image_512p = self.resize2_512p(image)
        image_tensor = process_images_([image_512p], self.image_processor)[0]

        image_size = image_512p.size
        mask = torch.tensor([1.0, 1.0, 1.0]) # 1 means using
        file_name = os.path.basename(input_img_path).split('.')[0]
        return dict(input_ids=input_ids.squeeze(),
                    image=image_tensor,
                    image_size=image_size,
                    file_name=file_name,
                    input_img = input_img,
                    retouch_mask=mask,
                    input_img_path=input_img_path,
                    )


class Infer_Param_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths=[],
        instruction_paths=[],
        model_path="./checkpoints/llava-fastvithd_0.5b_stage2",
        tokenizer=None,
        image_processor=None,
        samples_per_epoch=None,
        precision: str = "fp32",
        conv_mode: str = "qwen_2",
    ):
        super().__init__()
        self.conv_mode = conv_mode
        self.img_paths = img_paths
        self.instruction_paths = instruction_paths
        
        self.image_processor = image_processor
        self.precision = precision
        if tokenizer and image_processor:
            self.tokenizer = tokenizer
            self.image_processor = image_processor
        else:
            print("Lack of tokenizer or image processor.")

    def resize2_512p(self, image):
        target_short_edge = 512
        original_width, original_height = image.size
        w, h = original_width, original_height
        if w <= h:
            scale = target_short_edge / w
            target_w = target_short_edge
            target_h = int(h * scale)
        else:
            scale = target_short_edge / h
            target_h = target_short_edge
            target_w = int(w * scale)
        image_resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return image_resized
    
    def __len__(self):
        return len(self.img_paths)
    
    def parse_json_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print(f"错误：找不到文件 '{file_path}'")
            return None
        except json.JSONDecodeError as e:
            print(f"错误：文件格式不是有效的JSON。详细信息：{e}")
            return None
        except Exception as e:
            print(f"发生未知错误：{e}")
            return None 

    def get_organized_dict(self, json_dict):
        light_adj_dict = {
            "Exposure2012": 0.0,
            "Contrast2012": 0.0,
            "Highlights2012": 0.0,
            "Shadows2012": 0.0,
            "Whites2012": 0.0,
            "Blacks2012": 0.0,
            "ParametricShadows": 0.0,
            "ParametricDarks": 0.0,
            "ParametricLights": 0.0,
            "ParametricHighlights": 0.0,
        }
        colortemp_adj_dict = {
            "IncrementalTemperature": 0.0,
            "IncrementalTint": 0.0,
            "Vibrance": 0.0,
            "Saturation": 0.0,
        }
        colormixer_adj_dict = {
            "HueAdjustmentRed": 0.0,
            "HueAdjustmentOrange": 0.0,
            "HueAdjustmentYellow": 0.0,
            "HueAdjustmentGreen": 0.0,
            "HueAdjustmentAqua": 0.0,
            "HueAdjustmentBlue": 0.0,
            "HueAdjustmentPurple": 0.0,
            "HueAdjustmentMagenta": 0.0,
            "SaturationAdjustmentRed": 0.0,
            "SaturationAdjustmentOrange": 0.0,
            "SaturationAdjustmentYellow": 0.0,
            "SaturationAdjustmentGreen": 0.0,
            "SaturationAdjustmentAqua": 0.0,
            "SaturationAdjustmentBlue": 0.0,
            "SaturationAdjustmentPurple": 0.0,
            "SaturationAdjustmentMagenta": 0.0,
            "LuminanceAdjustmentRed": 0.0,
            "LuminanceAdjustmentOrange": 0.0,
            "LuminanceAdjustmentYellow": 0.0,
            "LuminanceAdjustmentGreen": 0.0,
            "LuminanceAdjustmentAqua": 0.0,
            "LuminanceAdjustmentBlue": 0.0,
            "LuminanceAdjustmentPurple": 0.0,
            "LuminanceAdjustmentMagenta": 0.0
        }

        key_list = list(json_dict.keys())
        light_key_list = list(light_adj_dict.keys())
        colortemp_key_list = list(colortemp_adj_dict.keys())
        colormixer_key_list = list(colormixer_adj_dict.keys())
        for key in key_list:
            if key in light_key_list:
                light_adj_dict[key] = json_dict[key]['value'] / 100.0
                light_adj_dict[key] = round(light_adj_dict[key], 3)
            elif key in colortemp_key_list:
                colortemp_adj_dict[key] = json_dict[key]['value'] / 100.0
                colortemp_adj_dict[key] = round(colortemp_adj_dict[key], 3)
            elif key in colormixer_key_list:
                colormixer_adj_dict[key] = json_dict[key]['value'] / 100.0
                colormixer_adj_dict[key] = round(colormixer_adj_dict[key], 3)
        new_dict = {
            "Light Adjustment": light_adj_dict,
            "Color and Temperature Adjustment": colortemp_adj_dict,
            "Specific Color Adjustment": colormixer_adj_dict
        }
        return new_dict

    def check_all_zero(self, data_dict):
        for value in data_dict.values():
            if value > 0.0001 or value < -0.0001:
                return False
        return True

    def get_mask_from_path(self, data_dict):
        # here, mask = 0 means in the future inference, the model would not see this control latent.
        if self.check_all_zero(data_dict["Light Adjustment"]):
            mask_1 = 0.0
        else:
            mask_1 = 1.0

        if self.check_all_zero(data_dict["Color and Temperature Adjustment"]):
            mask_2 = 0.0
        else:
            mask_2 = 1.0

        if self.check_all_zero(data_dict["Specific Color Adjustment"]):
            mask_3 = 0.0
        else:
            mask_3 = 1.0
        mask = torch.tensor([mask_1, mask_2, mask_3])
        return mask
    
    def __getitem__(self, idx):
        instruct_path = self.instruction_paths[idx]
        json_content = self.parse_json_file(instruct_path)
        retouch_params = self.get_organized_dict(json_content)
        mask = self.get_mask_from_path(retouch_params)

        qs = f"{TASK_PROFESSIONAL_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. I will provide an image and an professional instruction (plain text description or retouch operator parameters range from -1.0 to 1.0), please give me a retouch plan and retouch tokens.\n Instruction: {retouch_params}"

        conv = conv_templates[self.conv_mode].copy() #
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        input_img_path = self.img_paths[idx]
        input_img_ = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        input_img_tensor = transforms.ToTensor()(cv2.cvtColor(input_img_, cv2.COLOR_BGR2RGB))
        input_img = (input_img_tensor - 0.5) * 2
        # Load and preprocess image
        image = Image.open(input_img_path).convert('RGB')
        image_512p = self.resize2_512p(image)
        image_tensor = process_images_([image_512p], self.image_processor)[0]

        image_size = image_512p.size
        file_name = os.path.basename(input_img_path).split('.')[0]
        return dict(input_ids=input_ids.squeeze(),
                    image=image_tensor,
                    image_size=image_size,
                    file_name=file_name,
                    input_img = input_img,
                    retouch_mask=mask,
                    input_img_path=input_img_path,
                    )
            
class DataCollatorForUnifiedTestDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image_sizes'] = [instance['image_size'] for instance in instances]
            batch['images'] = images
        
        if 'input_img' in instances[0]:
            input_imgs = [instance['input_img'] for instance in instances]
            batch['input_imgs'] = input_imgs
        
        if 'target_img_path' in instances[0]:
            target_img_paths = [instance['target_img_path'] for instance in instances]
            batch['target_img_paths'] = target_img_paths
            
        if 'retouch_mask' in instances[0]:
            retouch_masks = [instance['retouch_mask'] for instance in instances]
            batch['retouch_masks'] = retouch_masks
        
        if 'input_img_path' in instances[0]:
            input_img_paths = [instance['input_img_path'] for instance in instances]
            batch['input_img_paths'] = input_img_paths
        return batch
