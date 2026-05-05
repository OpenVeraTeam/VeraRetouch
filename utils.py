from configs.renderer_config import load_config
from model.colormlp_v2 import ColorMLP_V2
from model.retouch_render import RetouchRenderer
import numpy as np
import json

def get_model(model_name, logger=None):
    model_config = load_config(model_name)
    if logger:
        logger.info(json.dumps(model_config, indent=2))
    if model_name == "colormlp_v2":
        model = ColorMLP_V2(model_config)
    # elif model_name in["RetouchRenderer_SiglipEncoder_noact", "RetouchRenderer_SiglipEncoderDual_DualAdaLN", "RetouchRenderer_Resnet22Encoder", "RetouchRenderer_SiglipEncoder", "RetouchRenderer_Resnet18Encoder", "RetouchRenderer_Resnet18_InteractEncoder", "RetouchRenderer_Resnet18_CrossInteractEncoder", "RetouchRenderer_SiglipEncoder_AdaIN", "RetouchRenderer_SiglipEncoder_Cat"]:
    elif 'RetouchRenderer' in model_name:
        model = RetouchRenderer(model_config)
    return model

def need_rerange(model_name):
    if model_name in ["colormlp"]:
        return True
    elif model_name in ["deep_preset", "colormlp_v2", "RetouchRenderer"]:
        return False
    
def tensor2img(img, dtype=None, input_scale=(0, 1)): 

    img = img.cpu().numpy()  
    a, b = input_scale 
    img = img.clip(a, b)
    img = (img-a)/(b - a) # [a, b] => [0, 1]
    img = (255.0*img).round()

    if len(img.shape) == 3: 
        img = img[None]
    img = img.transpose([0, 2, 3, 1]) 

    dtype = dtype if dtype else np.uint8
    img = img.astype(dtype) 
    if img.shape[0] == 1:
        img = img.squeeze(0)
    return img

def str2bool(v):
    """
        True: y, yes, t, true, on, 1
        False: n, no, f, false, off, 0
    """
    if isinstance(v, bool):
        return v
    if v.lower() in {"yes", "true", "t", "y", "1", "on"}:
        return True
    if v.lower() in {"no", "false", "f", "n", "0", "off"}:
        return False
    raise ValueError(f"Boolean value expected, got '{v}'")
