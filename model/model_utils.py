import numpy as np
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
    return img