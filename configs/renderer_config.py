# from asyncio.windows_events import NULL
from attrdict import AttrDict


def load_config(model_name):
    model_config = AttrDict({})
    if model_name == "deep_preset":
        model_config.g_depth = 5
        model_config.g_in_channels = 3
        model_config.g_norm = "evo"
        model_config.g_downsampler = "down_blurmax"
        model_config.g_upsampler = "up_blurbilinear"
        model_config.g_out_channels = 3
    elif model_name == "colormlp":
        model_config.num_filters = 50
    elif model_name == "colormlp_v2":
        model_config.hidden_dims = [128, 256, 512]
    elif model_name == "RetouchRenderer_SiglipEncoder":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "SiglipEncoder"
        encoder_config.latent_dim = 896
        encoder_config.num_tokens = None
        encoder_config.concat_dim = 1
        encoder_config.local_files_only = True
        encoder_config.lora_config = AttrDict({"r": 16, "lora_alpha": 16, "bias": "none", "target_modules": ["q_proj", "k_proj", "v_proj",  "in_proj", "out_proj"]})
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_SiglipEncoder_noact":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "SiglipEncoder"
        encoder_config.latent_dim = 896
        encoder_config.num_tokens = None
        encoder_config.concat_dim = 1
        encoder_config.local_files_only = True
        encoder_config.lora_config = AttrDict({"r": 16, "lora_alpha": 16, "bias": "none", "target_modules": ["q_proj", "k_proj", "v_proj",  "in_proj", "out_proj"]})
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "none"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_SiglipEncoder_hardsigmoid":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "SiglipEncoder"
        encoder_config.latent_dim = 896
        encoder_config.num_tokens = None
        encoder_config.concat_dim = 1
        encoder_config.local_files_only = True
        encoder_config.lora_config = AttrDict({"r": 16, "lora_alpha": 16, "bias": "none", "target_modules": ["q_proj", "k_proj", "v_proj",  "in_proj", "out_proj"]})
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "hard_sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_SiglipEncoder_hardtanh":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "SiglipEncoder"
        encoder_config.latent_dim = 896
        encoder_config.num_tokens = None
        encoder_config.concat_dim = 1
        encoder_config.local_files_only = True
        encoder_config.lora_config = AttrDict({"r": 16, "lora_alpha": 16, "bias": "none", "target_modules": ["q_proj", "k_proj", "v_proj",  "in_proj", "out_proj"]})
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "hard_tanh"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_SiglipEncoderDual_DualAdaLN":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "SiglipEncoderDual"
        encoder_config.latent_dim = 896
        encoder_config.num_tokens = None
        encoder_config.concat_dim = 1
        encoder_config.local_files_only = True
        encoder_config.lora_config = AttrDict({"r": 16, "lora_alpha": 16, "bias": "none", "target_modules": ["q_proj", "k_proj", "v_proj",  "in_proj", "out_proj"]})
        decoder_config.name = "ConditionalMLPDecoderDualAdaLN"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_SiglipEncoder_AdaIN":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "SiglipEncoder"
        encoder_config.latent_dim = 896
        encoder_config.num_tokens = None
        encoder_config.concat_dim = 1
        encoder_config.local_files_only = True
        encoder_config.lora_config = AttrDict({"r": 16, "lora_alpha": 16, "bias": "none", "target_modules": ["q_proj", "k_proj", "v_proj",  "in_proj", "out_proj"]})
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "adain"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_SiglipEncoder_Cat":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "SiglipEncoder"
        encoder_config.latent_dim = 896
        encoder_config.num_tokens = None
        encoder_config.concat_dim = 1
        encoder_config.local_files_only = True
        encoder_config.lora_config = AttrDict({"r": 16, "lora_alpha": 16, "bias": "none", "target_modules": ["q_proj", "k_proj", "v_proj",  "in_proj", "out_proj"]})
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "cat"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_VGG16Encoder":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "VGG16Encoder"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_VGG19Encoder":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "VGG19Encoder"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet22Encoder":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet22Encoder"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18_InteractEncoder":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18_InteractEncoder"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18_CrossInteractEncoder":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18_CrossInteractEncoder"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_CBAM":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_CBAM"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_MixedCBAM":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_MixedCBAM"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_MinusMixedCBAM":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_MinusMixedCBAM"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_2NetMinusMixedCBAM":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_2NetMinusMixedCBAM"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_InputRes":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_InputRes"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_SpecialAtten":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_SpecialAtten"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_InputCatMixedCBAM"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_3InputCatMixedCBAM":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_3InputCatMixedCBAM"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_FPN":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_FPN"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "RetouchRenderer_Resnet18Encoder_InputCatMixedCBAMFPN":
        encoder_config = AttrDict({})
        decoder_config = AttrDict({})
        encoder_config.model_name = "Resnet18Encoder_InputCatMixedCBAMFPN"
        encoder_config.latent_dim = 896
        decoder_config.name = "ConditionalMLPDecoder"
        decoder_config.latent_dim = 896
        decoder_config.hidden_dims = [128, 256, 512]
        decoder_config.cond_method = "add"
        decoder_config.final_act = "sigmoid"
        model_config.encoder_args = encoder_config
        model_config.decoder_args = decoder_config
    elif model_name == "Retouch_VQVAE_SIZE1024":
        model_config.latent_dim = 896
        model_config.book_size = 1024
        model_config.hidden_dims = [128, 256, 512]
        model_config.cond_method = "add"
        model_config.final_act = "sigmoid"
        model_config.vq_alpha = 0.01
        model_config.vq_beta = 0.25
    elif model_name == "Retouch_VQVAE_SIZE512":
        model_config.latent_dim = 896
        model_config.book_size = 512
        model_config.hidden_dims = [128, 256, 512]
        model_config.cond_method = "add"
        model_config.final_act = "sigmoid"
        model_config.vq_alpha = 0.01
        model_config.vq_beta = 0.25
    else:
        raise ValueError(f'没有这个模型：{model_name}')
    return model_config
        