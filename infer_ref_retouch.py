import os
import torch
import argparse
from utils import get_model
from PIL import Image
from utils import tensor2img
from torchvision import transforms
import cv2


def ref_img_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    preprocess = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((512, 512)),
        transforms.ToTensor(), 
    ])
    img = preprocess(img)
    img = (img - 0.5) * 2
    return img.unsqueeze(0)

def input_img_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transforms.ToTensor()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = (img - 0.5) * 2
    return img.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description='inference of encoder renderer')
    parser.add_argument('--pretrained_path', type=str, default="./checkpoints/encoder_renderer.pth", help='pretrained model path')
    parser.add_argument('--output_dir', type=str, default='./data_samples/ref_outputs', help='output dir path')
    parser.add_argument('--model_name', type=str, default='RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM')
    parser.add_argument('--input_img_path', type=str, default='./data_samples/ref_inputs/sample.jpg')
    parser.add_argument('--ref_input_img_path', type=str, default='./data_samples/ref_inputs/ref/before.jpg')
    parser.add_argument('--ref_target_img_path', type=str, default='./data_samples/ref_inputs/ref/after.jpg')
    parser.add_argument("--chunk", type=int, default=-1, help="Enable when GPU memory is insufficient. The renderer will process large images in chunks. Recommended value: 262144 (512*512)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = get_model(args.model_name)
    model.load_state_dict(torch.load(os.path.join(args.pretrained_path))["model_state_dict"])

    model.to(device)
    
    model.eval()
    with torch.no_grad():
        input_img = input_img_preprocess(args.input_img_path).to(device)
        ref_input_img = ref_img_preprocess(args.ref_input_img_path).to(device)
        ref_target_img = ref_img_preprocess(args.ref_target_img_path).to(device)
        mask = torch.tensor([1.0, 1.0, 1.0]).unsqueeze(0).to(device)
        save_name = os.path.basename(args.input_img_path).split('.')[0] + '.png'

        img_adj_pred = model(input_img, ref_input_img, ref_target_img, mask, args.chunk)
        img_adj_pred = img_adj_pred.clamp(0, 1)
        img_pil = tensor2img(img_adj_pred)
        img_pil = Image.fromarray(img_pil, mode='RGB')
        out_path = os.path.join(args.output_dir, save_name)
        img_pil.save(out_path, format='PNG')

if __name__ == "__main__":
    main()