import torch

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

CONTNET_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + '\n' + "Please describe the framing and content of this image.",
    DEFAULT_IMAGE_TOKEN + '\n' + "Can you provide a description of how this image is composed and what it depicts?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Could you outline both the structure and the subject matter of the photograph?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Please give an account of the image’s framing as well as its depicted elements.",
    DEFAULT_IMAGE_TOKEN + '\n' + "Tell me about the composition and the narrative content captured in this image.",
]

RETOUCH_TOKEN_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + '\n' + "Could you help me to retouch this image?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Can you assist with retouching this image for me?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Would you be able to help retouch this picture?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Could you do me a favor and retouch this image?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Is it possible for you to help with retouching this picture?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Can you lend a hand in retouching this image?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Would you help me out by retouching this picture?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Could you assist me in retouching this image?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Is there a way you could help retouch this picture for me?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Can you help me get this image retouched?",
    DEFAULT_IMAGE_TOKEN + '\n' + "Would you be willing to help with retouching this picture?",
]