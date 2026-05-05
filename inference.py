#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import sys
import os
import argparse
from utils import str2bool
import torch
from PIL import Image
import transformers
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.model.VeraRetouch import VeraRetouchForCausalLLM_Unified
from llava.constants import DEFAULT_RETOUCH_LIGHT_TOKEN, DEFAULT_RETOUCH_COLORTEMP_TOKEN, DEFAULT_RETOUCH_COLORMIXER_TOKEN
from llava.train.train_qwen_comment import find_all_linear_names
from safetensors.torch import load_file
import yaml
from box import Box
from utils import get_model as get_retouch_decoder_model
from torchvision import transforms
import cv2
from utils import tensor2img
from transformers import AutoTokenizer
from box import Box
from data.infer_dataset import Infer_Auto_Dataset, Infer_Style_Dataset, Infer_Param_Dataset, DataCollatorForUnifiedTestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def predict(args):
    model_path = os.path.expanduser(args.model_path)
    with open(args.config_add_path, "r", encoding="utf-8") as f:
        config_add = yaml.safe_load(f)
        config_add = Box(config_add)
        config_add.project_name = "test_no_instruct"
        config_add.freeze_retouch_decoder = True
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'),
                  generation_config)

    # Load model
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            cache_dir='./cache',
            model_max_length=args.max_new_tokens,
            padding_side="right",
            use_fast=False)
    model = VeraRetouchForCausalLLM_Unified.from_pretrained(
                model_path,
                config_add=config_add,
                cache_dir='./cache',
                torch_dtype=torch.bfloat16,
            ).cuda()

    retouch_token_light_idx = tokenizer(DEFAULT_RETOUCH_LIGHT_TOKEN, add_special_tokens=False).input_ids[0]
    retouch_token_colortemp_idx = tokenizer(DEFAULT_RETOUCH_COLORTEMP_TOKEN, add_special_tokens=False).input_ids[0]
    retouch_token_colormixer_idx = tokenizer(DEFAULT_RETOUCH_COLORMIXER_TOKEN, add_special_tokens=False).input_ids[0]
    model.register_special_token_idx(retouch_token_light_idx, retouch_token_colortemp_idx, retouch_token_colormixer_idx)
    image_processor = model.get_vision_tower().image_processor
    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    

    # 1) Dataset
    if args.mode == "auto":
        test_dataset = Infer_Auto_Dataset(img_paths = args.img_paths, tokenizer=tokenizer, image_processor=image_processor)
    elif args.mode == "style":
        test_dataset = Infer_Style_Dataset(img_paths = args.img_paths, prompts=args.prompts, tokenizer=tokenizer, image_processor=image_processor)
    elif args.mode == "param":
        test_dataset = Infer_Param_Dataset(img_paths = args.img_paths, instruction_paths=args.instruction_paths, tokenizer=tokenizer, image_processor=image_processor)
    
    # 2) Collator
    test_collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)

    # 3) DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=8,  
        collate_fn=test_collator,
    )

    count = 0
    for batch in tqdm(test_loader):
        input_img_paths = batch['input_img_paths']
        # if os.path.exists(img_save_path):
        #     continue

        input_ids = batch['input_ids'].cuda()
        images = batch.get('images', None)
        images = [tensor.to(torch.bfloat16).cuda() for tensor in images]
        attention_mask = batch['attention_mask'].cuda()
        input_imgs = batch['input_imgs']
        input_imgs = [tensor.to(torch.bfloat16).cuda() for tensor in input_imgs]
        image_sizes = batch['image_sizes']
        retouch_masks = batch['retouch_masks']
        retouch_masks = [tensor.to(torch.bfloat16).cuda() for tensor in retouch_masks]
        
        try:
            with torch.inference_mode():
                retouched_imgs, decoded_outputs = model._generate(tokenizer=tokenizer,
                                                                inputs=input_ids, 
                                                                attention_mask=attention_mask,
                                                                images=images, 
                                                                image_sizes=image_sizes,
                                                                retouch_masks=retouch_masks,
                                                                input_imgs=input_imgs,
                                                                do_sample=True if args.temperature > 0 else False,
                                                                temperature=args.temperature,
                                                                top_p=args.top_p,
                                                                num_beams=args.num_beams,
                                                                max_new_tokens=args.max_new_tokens,
                                                                output_hidden_states=True,
                                                                return_dict_in_generate=True,
                                                                chunk=args.chunk)  
            for serial, (retouch_img, decoded_output) in enumerate(zip(retouched_imgs, decoded_outputs)):
                img_name = os.path.basename(input_img_paths[serial]).split('.')[0]
                img_save_path = os.path.join(args.save_img_path, img_name+".png")
                text_save_path = os.path.join(args.save_text_path, img_name+".txt")
                cv2.imwrite(img_save_path, retouch_img)
                with open(text_save_path, 'w', encoding='utf-8') as f:
                    f.write(decoded_output)
                print(f"[{count}] Save image to {img_save_path} and text to {text_save_path}")
                count += 1
        except Exception as e:
            print(e)
            pass

    # Restore generation config
    if generation_config is not None:
        os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=f"/root/autodl-tmp/code/checkpoints/VeraRetouch")
    parser.add_argument("--config_add_path", type=str, default="./configs/infer_config.yaml")
    parser.add_argument("--mode", type=str, default="auto", help="You have to choose one of mode from auto, style and param")
    parser.add_argument("--img_paths", type=str, nargs='*', default=["./data_samples/input/sample_flower.jpg"], help="input image paths, multiple paths are supported")
    parser.add_argument("--prompt", type=str, default="I want a dreamy bright pink style.", help="style prompt(only 'style' mode used)")
    parser.add_argument("--instruction_path", type=str, default="./data_samples/param.json", help="retourch operator parameters(only 'param' mode used)")
    parser.add_argument("--save_dir", type=str, default=f"./data_samples/output/")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=-1, help="Enable when GPU memory is insufficient. The renderer will process large images in chunks. Recommended value: 262144 (512*512)")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    args = parser.parse_args()
       
    # check mode
    if args.mode in ["auto", "style", "param"]:
        print(f"You are using {args.mode} mode Now!")
    else:
        print(f"You must choose one mode from auto, style and param") 
        exit(0)

    # preprocess args
    args.save_img_path = os.path.join(args.save_dir, "pred_imgs")
    args.save_text_path = os.path.join(args.save_dir, "texts")
    os.makedirs(args.save_img_path, exist_ok=True)
    os.makedirs(args.save_text_path, exist_ok=True)

    if args.mode == "auto":
        pass
    elif args.mode == "style":
        args.prompts = [args.prompt for _ in args.img_paths]
    elif args.mode == "param":
        args.instruction_paths = [args.instruction_path for _ in args.img_paths]

    predict(args)
