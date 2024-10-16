# Based on https://huggingface.co/John6666/joy-caption-alpha-two-cli-modand https://github.com/chflame163/ComfyUI_LayerStyle

import os
import sys
import torch
from torch import nn
from typing import List, Union
from PIL import Image
import torchvision.transforms.functional as TVF
from torchvision.transforms import ToPILImage
import numpy as np
import folder_paths
import json
import logging
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import shutil
import gc
import comfy.model_management as mm

# Define the Joy2_Model class
class Joy2_Model:
    def __init__(self, clip_processor, clip_model, tokenizer, text_model, image_adapter):
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.image_adapter = image_adapter

# Define the ImageAdapter class
class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int,
                 deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)  # Matches HF's implementation of LLaMA

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.cat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (
            x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

# Define the model loading function
def load_models(model_path, dtype, device="cuda", device_map=None):
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
    from peft import PeftModel

    JC_lora = "text_model"
    use_lora = True if JC_lora != "none" else False
    CLIP_PATH = os.path.join(folder_paths.models_dir, "clip_vision", "google--siglip-so400m-patch14-384")
    CHECKPOINT_PATH = os.path.join(folder_paths.models_dir, "Joy_caption", "cgrkzexw-599808")
    LORA_PATH = os.path.join(CHECKPOINT_PATH, "text_model")

    if os.path.exists(CLIP_PATH):
        print("Start to load existing VLM")
    else:
        print("VLM not found locally. Downloading google/siglipso400m-patch14-384...")
        try:
            snapshot_download(
                repo_id="google/siglip-so400m-patch14-384", 
                local_dir=os.path.join(folder_paths.models_dir, "clip_vision", "cache--google--siglip-so400m-patch14-384"),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            shutil.move(os.path.join(folder_paths.models_dir, "clip_vision", "cache--google--siglip-so400m-patch14-384"), CLIP_PATH)
            print(f"VLM has been downloaded to {CLIP_PATH}")
        except Exception as e:
            print(f"Error downloading CLIP model: {e}")
            raise

    try:
        if dtype == "nf4":
            from transformers import BitsAndBytesConfig
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            print("Loading in NF4")
            print("Loading CLIP")
            clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
            clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model

            print("Loading VLM's custom vision model")
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, "clip_model.pt"), map_location='cpu', weights_only=False)
            checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
            clip_model.load_state_dict(checkpoint)
            del checkpoint
            clip_model.eval().requires_grad_(False).to(device)

            print("Loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(CHECKPOINT_PATH, "text_model"), use_fast=True)
            assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

            print(f"Loading LLM: {model_path}")
            text_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                quantization_config=nf4_config,
                device_map=device_map,  # 修改
                torch_dtype=torch.bfloat16
            ).eval()

            if use_lora and os.path.exists(LORA_PATH):  # omitted
                print("Loading VLM's custom text model")
                text_model = PeftModel.from_pretrained(
                    model=text_model, 
                    model_id=LORA_PATH, 
                    device_map=device if device == "cuda" else {"": device},
                    quantization_config=nf4_config
                )
                text_model = text_model.merge_and_unload(
                    safe_merge=True
                )  # to avoid PEFT bug https://github.com/huggingface/transformers/issues/28515
            else:
                print("VLM's custom text model isn't loaded")

            print("Loading image adapter")
            image_adapter = ImageAdapter(
                clip_model.config.hidden_size, 
                text_model.config.hidden_size, 
                False, False, 38,
                False
            ).eval().to("cpu")
            image_adapter.load_state_dict(
                torch.load(os.path.join(CHECKPOINT_PATH, "image_adapter.pt"), map_location=device, weights_only=False)
            )
            image_adapter.eval().to(device)
        else:  # bf16
            print("Loading in bfloat16")
            print("Loading CLIP")
            clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
            clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model
            if os.path.exists(os.path.join(CHECKPOINT_PATH, "clip_model.pt")):
                print("Loading VLM's custom vision model")
                checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, "clip_model.pt"), map_location=device, weights_only=False)
                checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
                clip_model.load_state_dict(checkpoint)
                del checkpoint
            clip_model.eval().requires_grad_(False).to(device)

            print("Loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(CHECKPOINT_PATH, "text_model"), use_fast=True)
            assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

            print(f"Loading LLM: {model_path}")
            text_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map=device_map,  # 修改
                torch_dtype=torch.bfloat16
            ).eval()

            if use_lora and os.path.exists(LORA_PATH):
                print("Loading VLM's custom text model")
                text_model = PeftModel.from_pretrained(
                    model=text_model, 
                    model_id=LORA_PATH, 
                    device_map=device if device == "cuda" else {"": device}
                )
                text_model = text_model.merge_and_unload(
                    safe_merge=True
                )  # to avoid PEFT bug https://github.com/huggingface/transformers/issues/28515
            else:
                print("VLM's custom text model isn't loaded")

            print("Loading image adapter")
            image_adapter = ImageAdapter(
                clip_model.config.hidden_size, 
                text_model.config.hidden_size, 
                False, False, 38,
                False
            ).eval().to(device)
            image_adapter.load_state_dict(
                torch.load(os.path.join(CHECKPOINT_PATH, "image_adapter.pt"), map_location=device, weights_only=False)
            )
    except Exception as e:
        print(f"Error loading models: {e}")
    finally:
        pass  # free_memory() function is not defined in the provided code

    return Joy2_Model(clip_processor, clip_model, tokenizer, text_model, image_adapter)

# Define the stream_chat function
@torch.inference_mode()
def stream_chat(input_images: List[Image.Image], caption_type: str, caption_length: Union[str, int],
                extra_options: list[str], name_input: str, custom_prompt: str,
                max_new_tokens: int, top_p: float, temperature: float, batch_size: int, model: Joy2_Model, device=str):

    CAPTION_TYPE_MAP = {
        "Descriptive": [
            "Write a descriptive caption for this image in a formal tone.",
            "Write a descriptive caption for this image in a formal tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a formal tone.",
        ],
        "Descriptive (Informal)": [
            "Write a descriptive caption for this image in a casual tone.",
            "Write a descriptive caption for this image in a casual tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a casual tone.",
        ],
        "Training Prompt": [
            "Write a stable diffusion prompt for this image.",
            "Write a stable diffusion prompt for this image within {word_count} words.",
            "Write a {length} stable diffusion prompt for this image.",
        ],
        "MidJourney": [
            "Write a MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
        ],
        "Booru tag list": [
            "Write a list of Booru tags for this image.",
            "Write a list of Booru tags for this image within {word_count} words.",
            "Write a {length} list of Booru tags for this image.",
        ],
        "Booru-like tag list": [
            "Write a list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
        ],
        "Art Critic": [
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
        ],
        "Product Listing": [
            "Write a caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
        ],
        "Social Media Post": [
            "Write a caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
        ],
    }

    all_captions = []

    # 'any' means no length specified
    length = None if caption_length == "any" else caption_length

    if isinstance(length, str):
        try:
            length = int(length)
        except ValueError:
            pass

    # Build prompt
    if length is None:
        map_idx = 0
    elif isinstance(length, int):
        map_idx = 1
    elif isinstance(length, str):
        map_idx = 2
    else:
        raise ValueError(f"Invalid caption length: {length}")

    prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

    # Add extra options
    if len(extra_options) > 0:
        prompt_str += " " + " ".join(extra_options)

    # Add name, length, word_count
    prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()

    # For debugging
    print(f"Prompt: {prompt_str}")

    for i in range(0, len(input_images), batch_size):
        batch = input_images[i:i + batch_size]

        for input_image in batch:
            try:
                # Preprocess image
                image = input_image.resize((384, 384), Image.LANCZOS)
                pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
                pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
                pixel_values = pixel_values.to(device)
            except ValueError as e:
                print(f"Error processing image: {e}")
                print("Skipping this image and continuing...")
                continue

            # Embed image
            with torch.amp.autocast_mode.autocast(device, enabled=True):
                vision_outputs = model.clip_model(pixel_values=pixel_values, output_hidden_states=True)
                image_features = vision_outputs.hidden_states
                embedded_images = model.image_adapter(image_features).to(device)

            # Build the conversation
            convo = [
                {
                    "role": "system",
                    "content": "You are a helpful image captioner.",
                },
                {
                    "role": "user",
                    "content": prompt_str,
                },
            ]

            # Format the conversation
            if hasattr(model.tokenizer, 'apply_chat_template'):
                convo_string = model.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback if apply_chat_template is not available
                convo_string = "<|eot_id|>\n"
                for message in convo:
                    if message['role'] == 'system':
                        convo_string += f"<|system|>{message['content']}<|endoftext|>\n"
                    elif message['role'] == 'user':
                        convo_string += f"<|user|>{message['content']}<|endoftext|>\n"
                    else:
                        convo_string += f"{message['content']}<|endoftext|>\n"
                convo_string += "<|eot_id|>"

            assert isinstance(convo_string, str)

            # Tokenize the conversation
            convo_tokens = model.tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False,
                                                  truncation=False)
            prompt_tokens = model.tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False,
                                                   truncation=False)
            assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
            convo_tokens = convo_tokens.squeeze(0)
            prompt_tokens = prompt_tokens.squeeze(0)

            # Calculate where to inject the image
            eot_id_indices = (convo_tokens == model.tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[
                0].tolist()
            assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

            preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]

            # Embed the tokens
            convo_embeds = model.text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))

            # Construct the input
            input_embeds = torch.cat([
                convo_embeds[:, :preamble_len],
                embedded_images.to(dtype=convo_embeds.dtype),
                convo_embeds[:, preamble_len:],
            ], dim=1).to(device)

            input_ids = torch.cat([
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                convo_tokens[preamble_len:].unsqueeze(0),
            ], dim=1).to(device)
            attention_mask = torch.ones_like(input_ids)

            generate_ids = model.text_model.generate(input_ids=input_ids, inputs_embeds=input_embeds,
                                                     attention_mask=attention_mask, do_sample=True,
                                                     suppress_tokens=None, max_new_tokens=max_new_tokens, top_p=top_p,
                                                     temperature=temperature)

            # Trim off the prompt
            generate_ids = generate_ids[:, input_ids.shape[1]:]
            if generate_ids[0][-1] == model.tokenizer.eos_token_id or generate_ids[0][-1] == model.tokenizer.convert_tokens_to_ids(
                    "<|eot_id|>"):
                generate_ids = generate_ids[:, :-1]

            caption = model.tokenizer.batch_decode(generate_ids, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)[0]
            all_captions.append(caption.strip())

    return all_captions

def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
def take_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()
    elif isinstance(dev, str):
        if dev.startswith('cuda'):
            # 如果设备是 'cuda'，则默认使用索引 0
            if ':' not in dev:
                dev = torch.device('cuda:0')
            else:
                dev = torch.device(dev)
        else:
            dev = torch.device(dev)
    elif not isinstance(dev, torch.device):
        dev = torch.device(dev)

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024  # TODO: 实现 DirectML 显存检测
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total
        
def cleanGPU():
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache()

def analyze_tensor(tensor: object) -> str:
    result = ''
    if isinstance(tensor, torch.Tensor):
        result += f"\n Dimensions: {tensor.dim()}, First dimension size: {tensor.shape[0]} \n"
        for idx, t in enumerate(tensor):
            image = Image.fromarray(np.clip(255.0 * t.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            result += f'\n Index {idx}: Image dimensions = {image.size}, Mode = {image.mode}, Tensor dims = {t.dim()}, '
            result += ', '.join([f'Dimension {j} size = {t.shape[j]}' for j in range(t.dim())])
    else:
        result = f"analyze_tensor: Input is not a tensor, found {type(tensor)} instead"
    return result
            

class JoyCaption2:
    
    CATEGORY = 'TTP_Toolset'
    FUNCTION = "joycaption2"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)
    
    def __init__(self):
        self.NODE_NAME = 'JoyCaption2'
        self.previous_model = None

    @classmethod
    def INPUT_TYPES(cls):
        llm_model_list = ["unsloth/Meta-Llama-3.1-8B-Instruct", "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"]
        dtype_list = ['nf4', 'bf16']
        caption_type_list = [
            "Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney",
            "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing",
            "Social Media Post"
        ]
        caption_length_list = ["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 5)]
        
        # get extra_option.json path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        extra_option_file = os.path.join(base_dir, "extra_option.json") 

        # load extra_options_list
        extra_options_list = {}
        if os.path.isfile(extra_option_file):
            try:
                with open(extra_option_file, "r", encoding='utf-8') as f:
                    json_content = json.load(f)
                    for item in json_content:
                        option_name = item.get("name")
                        if option_name:
                            extra_options_list[option_name] = ("BOOLEAN", {"default": False})
                            # logger.info(f"Loaded extra option: {option_name}")
            except Exception as e:
                print(f"Error loading extra_option.json: {e}")
        else:
            print(f"extra_option.json not found at {extra_option_file}. No extra options will be available.")

        # 定义额外的输入字段
        return {
            "required": {
                "image": ("IMAGE",),
                "llm_model": (llm_model_list,),
                "dtype": (dtype_list,),
                "caption_type": (caption_type_list,),
                "caption_length": (caption_length_list,),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 260, "min": 8, "max": 4096, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "cache_model": ("BOOLEAN", {"default": False}),
                "enable_extra_options": ("BOOLEAN", {"default": True, "label": "启用额外选项"}),  # 新增开关
                **extra_options_list,  
                "character_name": ("STRING", {"default": "", "multiline": False}),  
            },
        }
 

    def joycaption2(
        self, image, llm_model, dtype, caption_type, caption_length,
        user_prompt, max_new_tokens, top_p, temperature, cache_model,
        enable_extra_options, character_name, **extra_options  
    ):
        ret_text = [] 
        comfy_model_dir = os.path.join(folder_paths.models_dir, "LLM")
        print(f"comfy_model_dir:{comfy_model_dir}")
        if not os.path.exists(comfy_model_dir):
            os.mkdir(comfy_model_dir)
        
        sanitized_model_name = llm_model.replace('/', '--')
        llm_model_path = os.path.join(comfy_model_dir, sanitized_model_name)  
        llm_model_path_cache = os.path.join(comfy_model_dir, "cache--" + sanitized_model_name)

        # 初始设备设置为 'cuda'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_loaded_on = device  # 跟踪模型加载在哪个设备上

        try:
            if os.path.exists(llm_model_path):
                print(f"Start to load existing model on {device}")
            else:
                print(f"Model not found locally. Downloading {llm_model}...")
                snapshot_download(
                    repo_id=llm_model, 
                    local_dir=llm_model_path_cache, 
                    local_dir_use_symlinks=False, 
                    resume_download=True
                )
                shutil.move(llm_model_path_cache, llm_model_path)   
                print(f"Model downloaded to {llm_model_path}...")
            
            if self.previous_model is None:
                try:
                    # 尝试加载模型
                    free_vram_bytes = mm.get_free_memory()
                    free_vram_gb = free_vram_bytes / (1024 ** 3)
                    print(f"Free VRAM: {free_vram_gb:.2f} GB")
                    if dtype == 'nf4' and free_vram_gb < 10:
                        print("Free VRAM is less than 10GB when loading 'nf4' model. Performing VRAM cleanup.")
                        cleanGPU()
                    elif dtype == 'bf16' and free_vram_gb < 20:
                        print("Free VRAM is less than 20GB when loading 'bf16' model. Performing VRAM cleanup.")
                        cleanGPU()                    
                    device_map = "auto"
                    model = load_models(
                        model_path=llm_model_path, dtype=dtype, device=device,
                        device_map=device_map
                    )
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print("显存不足，正在尝试使用共享显存...")
                        model = load_models(
                            model_path=llm_model_path, dtype=dtype, device=device,
                            device_map="auto"
                        )
                        print("模型已使用共享显存加载。")
                        raise e
                    else:
                        raise e
            else:
                model = self.previous_model

        except Exception as e:
            print(f"Error loading model: {e}")
            return ("Error loading model.",)

        print(f"Model loaded on {model_loaded_on}")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        extra_option_file = os.path.join(base_dir, "extra_option.json")  # 调整为 JSON 文件的正确路径
        extra_prompts = {}

        if enable_extra_options and os.path.isfile(extra_option_file):
            try:
                with open(extra_option_file, "r", encoding='utf-8') as f:
                    json_content = json.load(f)
                    for item in json_content:
                        name = item.get("name")
                        prompt = item.get("prompt")
                        if name and prompt:
                            extra_prompts[name] = prompt
            except Exception as e:
                print(f"Error reading extra_option.json: {e}")
        elif not os.path.isfile(extra_option_file):
            print(f"extra_option.json not found at {extra_option_file} during processing.")

        extra = []
        if enable_extra_options:
            for option_name, is_enabled in extra_options.items():
                if is_enabled and option_name in extra_prompts:
                    extra.append(extra_prompts[option_name])

            processed_images = [
                Image.fromarray(
                    np.clip(255.0 * img.unsqueeze(0).cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                ).convert('RGB')
                for img in image
            ]

            captions = stream_chat(
                processed_images, caption_type, caption_length,
                extra, "", user_prompt,
                max_new_tokens, top_p, temperature, len(processed_images),
                model, device
            )

            ret_text.extend(captions)

        if cache_model:
            self.previous_model = model
        else:
            self.previous_model = None
            del model
            free_memory()

        return (ret_text,)


class ExtraOptionsNode:
    CATEGORY = 'TTP_Toolset'
    FUNCTION = "extra_options"
    RETURN_TYPES = ("STRING",)  # 改为返回单一字符串
    RETURN_NAMES = ("extra_options_str",)
    OUTPUT_IS_LIST = (False,)  # 单一字符串输出

    def __init__(self):
        self.NODE_NAME = 'ExtraOptionsNode'

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 extra_option.json 的路径并加载选项
        base_dir = os.path.dirname(os.path.abspath(__file__))
        extra_option_file = os.path.join(base_dir, "extra_option.json")
        extra_options_list = {}

        if os.path.isfile(extra_option_file):
            try:
                with open(extra_option_file, "r", encoding='utf-8') as f:
                    json_content = json.load(f)
                    for item in json_content:
                        option_name = item.get("name")
                        if option_name:
                            # 定义每个额外选项为布尔输入
                            extra_options_list[option_name] = ("BOOLEAN", {"default": False})
            except Exception as e:
                print(f"Error loading extra_option.json: {e}")
        else:
            print(f"extra_option.json not found at {extra_option_file}. No extra options will be available.")

        # 定义输入字段，包括开关和 character_name
        return {
            "required": {
                "enable_extra_options": ("BOOLEAN", {"default": True, "label": "启用额外选项"}),  # 开关
                **extra_options_list,  # 动态加载的额外选项
                "character_name": ("STRING", {"default": "", "multiline": False}),  # 移动 character_name
            },
        }

    def extra_options(self, enable_extra_options, character_name, **extra_options):
        """
        处理额外选项并返回已启用的提示列表。
        如果启用了替换角色名称选项，并提供了 character_name，则进行替换。
        """
        extra_prompts = []
        if enable_extra_options:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            extra_option_file = os.path.join(base_dir, "extra_option.json")
            if os.path.isfile(extra_option_file):
                try:
                    with open(extra_option_file, "r", encoding='utf-8') as f:
                        json_content = json.load(f)
                        for item in json_content:
                            name = item.get("name")
                            prompt = item.get("prompt")
                            if name and prompt:
                                if extra_options.get(name):
                                    # 如果 prompt 中包含 {name}，则替换为 character_name
                                    if "{name}" in prompt:
                                        prompt = prompt.replace("{name}", character_name)
                                    extra_prompts.append(prompt)
                except Exception as e:
                    print(f"Error reading extra_option.json: {e}")
            else:
                print(f"extra_option.json not found at {extra_option_file} during processing.")

        # 将所有启用的提示拼接成一个字符串
        return (" ".join(extra_prompts),)  # 返回一个单一的合并字符串

class JoyCaption2_simple:
    
    CATEGORY = 'TTP_Toolset'
    FUNCTION = "joycaption2_simple"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)
    
    def __init__(self):
        self.NODE_NAME = 'JoyCaption2_simple'
        self.previous_model = None

    @classmethod
    def INPUT_TYPES(cls):
        llm_model_list = [
            "unsloth/Meta-Llama-3.1-8B-Instruct",
            "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
        ]
        dtype_list = ['nf4', 'bf16']
        caption_type_list = [
            "Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney",
            "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing",
            "Social Media Post"
        ]
        caption_length_list = [
            "any", "very short", "short", "medium-length", "long", "very long"
        ] + [str(i) for i in range(20, 261, 5)]

        # 定义额外的输入字段
        return {
            "required": {
                "image": ("IMAGE",),
                "llm_model": (llm_model_list,),
                "dtype": (dtype_list,),
                "caption_type": (caption_type_list,),
                "caption_length": (caption_length_list,),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 260, "min": 8, "max": 4096, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "cache_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "extra_options_node": ("STRING",{"forceInput": True}),  # 接收来自 ExtraOptionsNode 的单一字符串
            },    
        }

    def joycaption2_simple(
        self, image, llm_model, dtype, caption_type, caption_length,
        user_prompt, max_new_tokens, top_p, temperature, cache_model, 
        extra_options_node=None  # 设置默认值为 None
    ):
        ret_text = [] 
        comfy_model_dir = os.path.join(folder_paths.models_dir, "LLM")
        print(f"comfy_model_dir: {comfy_model_dir}")
        if not os.path.exists(comfy_model_dir):
            os.mkdir(comfy_model_dir)
        
        sanitized_model_name = llm_model.replace('/', '--')
        llm_model_path = os.path.join(comfy_model_dir, sanitized_model_name)  
        llm_model_path_cache = os.path.join(comfy_model_dir, "cache--" + sanitized_model_name)

        # 初始设备设置为 'cuda'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_loaded_on = device  # 跟踪模型加载在哪个设备上

        try:
            if os.path.exists(llm_model_path):
                print(f"Start to load existing model on {device}")
            else:
                print(f"Model not found locally. Downloading {llm_model}...")
                snapshot_download(
                    repo_id=llm_model, 
                    local_dir=llm_model_path_cache, 
                    local_dir_use_symlinks=False, 
                    resume_download=True
                )
                shutil.move(llm_model_path_cache, llm_model_path)   
                print(f"Model downloaded to {llm_model_path}...")
            
            if self.previous_model is None:
                try:
                    # 尝试加载模型
                    free_vram_bytes = mm.get_free_memory()
                    free_vram_gb = free_vram_bytes / (1024 ** 3)
                    print(f"Free VRAM: {free_vram_gb:.2f} GB")
                    if dtype == 'nf4' and free_vram_gb < 10:
                        print("Free VRAM is less than 10GB when loading 'nf4' model. Performing VRAM cleanup.")
                        cleanGPU()
                    elif dtype == 'bf16' and free_vram_gb < 20:
                        print("Free VRAM is less than 20GB when loading 'bf16' model. Performing VRAM cleanup.")
                        cleanGPU()                    
                    device_map = "auto"
                    model = load_models(
                        model_path=llm_model_path, dtype=dtype, device=device,
                        device_map=device_map
                    )
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print("显存不足，正在尝试使用共享显存...")
                        model = load_models(
                            model_path=llm_model_path, dtype=dtype, device=device,
                            device_map="auto"
                        )
                        print("模型已使用共享显存加载。")
                        raise e
                    else:
                        raise e
            else:
                model = self.previous_model

        except Exception as e:
            print(f"Error loading model: {e}")
            return ("Error loading model.",)

        print(f"Model loaded on {model_loaded_on}")

        # 接收来自 ExtraOptionsNode 的额外提示
        extra = []
        if extra_options_node and extra_options_node.strip():
            extra = [extra_options_node]  # 将单一字符串包装成列表
            print(f"Extra options enabled: {extra_options_node}")
        else:
            print("No extra options provided.")

        # 处理图像
        processed_images = [
            Image.fromarray(
                np.clip(255.0 * img.unsqueeze(0).cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            ).convert('RGB')
            for img in image
        ]

        try:
            captions = stream_chat(
                processed_images, caption_type, caption_length,
                extra, "", user_prompt,
                max_new_tokens, top_p, temperature, len(processed_images),
                model, device  # 确保传递正确的设备
            )
            ret_text.extend(captions)
        except Exception as e:
            print(f"Error during stream_chat: {e}")
            return ("Error generating captions.",)

        if cache_model:
            self.previous_model = model
        else:
            self.previous_model = None
            del model
            free_memory()

        return (ret_text,)
        
# Register the node
NODE_CLASS_MAPPINGS = {
    "JoyCaption2": JoyCaption2,
    "ExtraOptionsNode": ExtraOptionsNode,
    "JoyCaption2_simple": JoyCaption2_simple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyCaption2": "TTP_JoyCaption2_Full",
    "ExtraOptionsNode": "TTP_ExtraOptionsNode",
    "JoyCaption2_simple": "TTP_JoyCaption2_simple",
}
