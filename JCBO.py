import os
import sys
import torch
from torch import nn
from typing import List, Union, Generator
from PIL import Image
import torchvision.transforms.functional as TVF
import numpy as np
import folder_paths
import json
import logging
from transformers import LlavaForConditionalGeneration, TextIteratorStreamer, AutoProcessor, BitsAndBytesConfig
from huggingface_hub import snapshot_download
import shutil
import gc
import comfy.model_management as mm
import comfy.sd
from threading import Thread

# LIGER Kernel import attempt
try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False
    print("LIGER kernel not found. The option to enable it will be disabled.")

# Global model cache
CACHED_MODEL = None
CACHED_PROCESSOR = None
CACHED_MODEL_PATH_HF_ID = None # Stores the HuggingFace model ID used for the cache
CACHED_LIGER_ENABLED = None
CACHED_QUANTIZATION_MODE = None
CACHED_MODEL_LOCAL_PATH = None # Stores the local disk path of the cached model

QUANTIZATION_CONFIGS = {
    "nf4": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
        },
    "int8": {
        "load_in_8bit": True,
    },
    "bf16": { # bf16 is not a quantization config, but a torch_dtype. Handled separately.
    },
}
LLM_SKIP_MODULES = ["vision_tower", "multi_modal_projector"]
MODEL_PATH_HF_DEFAULT = "fancyfeast/llama-joycaption-beta-one-hf-llava"

# Define the CAPTION_TYPE_MAP for JoyCaptionBetaOne
CAPTION_TYPE_MAP_BETA = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        '''Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what\'s absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.''',
        '''Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what\'s absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.''',
        '''Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what\'s absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.''',
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by \'artist:\', \'copyright:\', \'character:\', \'species:\', \'meta:\', and \'lore:\'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by \'artist:\', \'copyright:\', \'character:\', \'species:\', \'meta:\', and \'lore:\'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by \'artist:\', \'copyright:\', \'character:\', \'species:\', \'meta:\', and \'lore:\'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by \'artist:\', \'copyright:\', \'character:\', and \'meta:\'. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by \'artist:\', \'copyright:\', \'character:\', and \'meta:\'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by \'artist:\', \'copyright:\', \'character:\', and \'meta:\'. Then all the general tags.",
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

NAME_OPTION_PROMPT = "If there is a person/character in the image you must refer to them as {name}."

# Extra options for the node
EXTRA_OPTIONS_LIST = [
    NAME_OPTION_PROMPT,
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image\'s resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image\'s composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
    "If it is a work of art, do not include the artist\'s name or the title of the work.",
    "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
    """Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.""",
    "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
    "Include information about the ages of any people/characters when applicable.",
    "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
    "Do not mention the mood/feeling/etc of the image.",
    "Explicitly specify the vantage height (eye-level, low-angle worm's-eye, bird's-eye, drone, rooftop, etc.).",
    "If there is a watermark, you must mention it.",
    """Your response will be used by a text-to-image model, so avoid useless meta phrases like "This image shows…", "You are looking at...", etc.""",
]

def _build_prompt_beta_shared(caption_type: str, caption_length: str, list_of_extra_prompts: list[str], character_name_val: str, custom_prompt_override: str) -> str:
    if custom_prompt_override and custom_prompt_override.strip():
        if character_name_val and "{name}" in custom_prompt_override:
            return custom_prompt_override.replace("{name}", character_name_val)
        return custom_prompt_override

    if caption_length == "any": map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit(): map_idx = 1
    else: map_idx = 2
    
    base_prompt = CAPTION_TYPE_MAP_BETA[caption_type][map_idx]
    final_extra_prompts = []

    for extra_prompt_template in list_of_extra_prompts:
        if extra_prompt_template == NAME_OPTION_PROMPT:
            if character_name_val: # Only include and format name if provided
                final_extra_prompts.append(extra_prompt_template.format(name=character_name_val))
            # If character_name_val is empty, this prompt is skipped entirely.
        else:
            final_extra_prompts.append(extra_prompt_template)
        
    full_prompt_parts = [base_prompt]
    if final_extra_prompts:
        full_prompt_parts.extend(final_extra_prompts)
    
    # Format the base prompt part; extra prompts are already formatted or don't need it here
    # This assumes {word_count} and {length} are only in the base_prompt template
    full_prompt_parts[0] = full_prompt_parts[0].format(length=caption_length, word_count=caption_length)
    
    return " ".join(full_prompt_parts)

def _free_model_memory_shared():
    global CACHED_MODEL, CACHED_PROCESSOR, CACHED_MODEL_PATH_HF_ID, CACHED_LIGER_ENABLED, CACHED_QUANTIZATION_MODE, CACHED_MODEL_LOCAL_PATH
    CACHED_MODEL = None
    CACHED_PROCESSOR = None
    CACHED_MODEL_PATH_HF_ID = None
    CACHED_LIGER_ENABLED = None
    CACHED_QUANTIZATION_MODE = None
    CACHED_MODEL_LOCAL_PATH = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("JoyCaptionBetaOne (Shared): Model and processor released from cache.")

def _clean_gpu_shared():
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache()
    print("JoyCaptionBetaOne (Shared): ComfyUI models unloaded and cache soft-emptied.")

def _load_model_shared(model_hf_id: str, quantization_mode: str, target_device: str, enable_liger: bool):
    global CACHED_MODEL, CACHED_PROCESSOR, CACHED_MODEL_PATH_HF_ID, CACHED_LIGER_ENABLED, CACHED_QUANTIZATION_MODE, CACHED_MODEL_LOCAL_PATH

    model_dir_base = os.path.join(folder_paths.models_dir, "LLM_llava")
    if not os.path.exists(model_dir_base): os.makedirs(model_dir_base, exist_ok=True)
    sanitized_model_repo_name = model_hf_id.replace('/', '--')
    model_path_local = os.path.join(model_dir_base, sanitized_model_repo_name)
    model_path_cache_tmp = os.path.join(model_dir_base, "cache--" + sanitized_model_repo_name)

    effective_device = target_device if torch.cuda.is_available() else "cpu"
    print(f"JoyCaptionBetaOne (Shared): Using effective device: {effective_device} for model {model_hf_id}")

    reload_needed = False
    if CACHED_MODEL is None or \
       CACHED_PROCESSOR is None or \
       CACHED_MODEL_PATH_HF_ID != model_hf_id or \
       CACHED_MODEL_LOCAL_PATH != model_path_local or \
       CACHED_QUANTIZATION_MODE != quantization_mode or \
       (LIGER_KERNEL_AVAILABLE and CACHED_LIGER_ENABLED != enable_liger):
        reload_needed = True
        if CACHED_MODEL is not None:
            print(f"JoyCaptionBetaOne (Shared): Config changed (Prev: {CACHED_MODEL_PATH_HF_ID}, {CACHED_QUANTIZATION_MODE}, Liger: {CACHED_LIGER_ENABLED}. New: {model_hf_id}, {quantization_mode}, Liger: {enable_liger}). Reloading.")
            _free_model_memory_shared()

    if reload_needed:
        print(f"JoyCaptionBetaOne (Shared): Loading model from {model_path_local} (HF: {model_hf_id})")
        if not os.path.exists(model_path_local):
            print(f"JoyCaptionBetaOne (Shared): Downloading {model_hf_id}...")
            try:
                snapshot_download(repo_id=model_hf_id, local_dir=model_path_cache_tmp, local_dir_use_symlinks=False, resume_download=True)
                shutil.move(model_path_cache_tmp, model_path_local)
                print(f"JoyCaptionBetaOne (Shared): Model {model_hf_id} downloaded to {model_path_local}")
            except Exception as e: raise RuntimeError(f"Error downloading model {model_hf_id}: {e}")
        try:
            print(f"JoyCaptionBetaOne (Shared): Loading processor from {model_path_local}...")
            processor = AutoProcessor.from_pretrained(model_path_local)
            print(f"JoyCaptionBetaOne (Shared): Loading model {model_hf_id} with quantization '{quantization_mode}'...")
            
            model_load_kwargs = {}
            final_torch_dtype = None
            final_device_map = "auto" # Default to auto, will be overridden if needed

            current_quant_mode = quantization_mode
            if "cuda" not in effective_device and current_quant_mode in ["nf4", "int8"]:
                print(f"JoyCaptionBetaOne (Shared): Quantization '{current_quant_mode}' needs CUDA. Falling back to bf16 for CPU for {model_hf_id}.")
                current_quant_mode = "bf16"
            
            if current_quant_mode == "bf16":
                final_torch_dtype = torch.bfloat16
                final_device_map = None if "cpu" in effective_device else effective_device
            elif current_quant_mode in ["nf4", "int8"]:
                # This block is for CUDA devices as per the check above
                bnb_config_params = QUANTIZATION_CONFIGS[current_quant_mode].copy()
                bnb_config_params["llm_int8_skip_modules"] = LLM_SKIP_MODULES
                q_config = BitsAndBytesConfig(**bnb_config_params)
                model_load_kwargs["quantization_config"] = q_config
                final_torch_dtype = torch.bfloat16 if current_quant_mode == "nf4" else "auto"
                final_device_map = effective_device # MODIFICATION: Use the user-selected CUDA device
                print(f"JoyCaptionBetaOne (Shared): Preparing {current_quant_mode} for specific device: {effective_device}")
            else: # Fallback / fp32 (though not an explicit option)
                final_torch_dtype = torch.float32 if "cpu" in effective_device else torch.bfloat16
                final_device_map = None if "cpu" in effective_device else effective_device

            model_load_kwargs["torch_dtype"] = final_torch_dtype
            model_load_kwargs["device_map"] = final_device_map
            
            if "cuda" in effective_device:
                free_vram_gb = mm.get_free_memory(effective_device) / (1024**3)
                # Basic VRAM check - can be more sophisticated
                if free_vram_gb < 4 and current_quant_mode != "nf4": # NF4 is very light
                     print(f"Warning: Low VRAM ({free_vram_gb:.2f}GB on {effective_device}) for {current_quant_mode}")
                     # _clean_gpu_shared() # Consider if cleanup is aggressive enough or needed
            
            model = LlavaForConditionalGeneration.from_pretrained(model_path_local, **model_load_kwargs)
            assert isinstance(model, LlavaForConditionalGeneration)
            model.eval()

            if LIGER_KERNEL_AVAILABLE and enable_liger and "cuda" in str(model.device).lower(): # Check actual model device for LIGER
                try:
                    print(f"JoyCaptionBetaOne (Shared): Applying LIGER kernel to {model_hf_id} on {model.device}...")
                    apply_liger_kernel_to_llama(model=model.language_model)
                    CACHED_LIGER_ENABLED = True
                except Exception as e: print(f"JoyCaptionBetaOne (Shared): LIGER kernel apply failed for {model_hf_id}: {e}"); CACHED_LIGER_ENABLED = False
            else: CACHED_LIGER_ENABLED = False
            
            CACHED_MODEL = model
            CACHED_PROCESSOR = processor
            CACHED_MODEL_PATH_HF_ID = model_hf_id
            CACHED_MODEL_LOCAL_PATH = model_path_local
            CACHED_QUANTIZATION_MODE = quantization_mode # Cache the original requested mode
            print(f"JoyCaptionBetaOne (Shared): Model {model_hf_id} loaded. Effective quantization: '{current_quant_mode}', LIGER: {CACHED_LIGER_ENABLED}, Device map: '{str(model.hf_device_map)}'.")
        except Exception as e: 
            _free_model_memory_shared()
            raise RuntimeError(f"Error loading model {model_hf_id}: {e}")
    else:
        print(f"JoyCaptionBetaOne (Shared): Using cached model ({CACHED_MODEL_PATH_HF_ID}, Quant: {CACHED_QUANTIZATION_MODE}, LIGER: {CACHED_LIGER_ENABLED}).")
        model = CACHED_MODEL
        processor = CACHED_PROCESSOR
    return model, processor

class JoyCaptionBetaOne_Full:
    CATEGORY = 'TTP_Toolset'
    FUNCTION = "caption_image"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    OUTPUT_IS_LIST = (True,)
    
    def __init__(self):
        self.NODE_NAME = 'JoyCaptionBetaOne_Full'

    @classmethod
    def INPUT_TYPES(cls):
        caption_type_keys = list(CAPTION_TYPE_MAP_BETA.keys())
        caption_length_list = ["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)]
        quantization_mode_list = ['bf16', 'nf4', 'int8'] 
        
        gpu_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not gpu_devices:
            gpu_devices = ["cpu"]

        extra_options_inputs = {}
        for i, option_text in enumerate(EXTRA_OPTIONS_LIST):
            label = option_text.split('.')[0].replace(' ', '_').replace('/', '_').lower()
            if len(label) > 30: label = label[:30]
            extra_options_inputs[f"extra_option_{i}_{label}"] = ("BOOLEAN", {"default": False, "label": option_text[:100]})

        inputs = {
            "required": {
                "image": ("IMAGE",),
                "caption_type": (caption_type_keys,),
                "caption_length": (caption_length_list,),
                "quantization_mode": (quantization_mode_list, {"default": 'bf16'}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "label": "Custom Prompt (Overrides caption type & extras)"}),
                "character_name": ("STRING", {"default": "", "multiline": False, "label": "Person/Character Name (for {name} in extras)"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "device": (gpu_devices,),
                "cache_model": ("BOOLEAN", {"default": True, "label": "Cache Model in Memory"}),
            },
            "optional": {} 
        }
        
        if LIGER_KERNEL_AVAILABLE:
            inputs["required"]["enable_liger_kernel"] = ("BOOLEAN", {"default": True, "label": "Enable LIGER Kernel (CUDA only)"})
        else:
            inputs["required"]["info_liger_unavailable"] = ("STRING", {"default": "LIGER Kernel not installed/available.", "label": "LIGER Kernel Status", "input": "hidden"})

        # Add the dynamically generated extra options to 'required'
        inputs["required"].update(extra_options_inputs)
        return inputs

    @torch.no_grad()
    def caption_image(self, image: torch.Tensor, caption_type: str, caption_length: str,
                      quantization_mode: str, 
                      custom_prompt: str, character_name: str,
                      temperature: float, top_p: float, max_new_tokens: int,
                      device: str, cache_model: bool, **kwargs):
        enable_liger_kernel = kwargs.get('enable_liger_kernel', False) if LIGER_KERNEL_AVAILABLE else False
        try:
            model, processor = _load_model_shared(MODEL_PATH_HF_DEFAULT, quantization_mode, device, enable_liger_kernel)
        except Exception as e:
            print(f"Error in {self.NODE_NAME}: {e}")
            return ([str(e)],) # Return error message as list of strings
        
        selected_extra_options_prompts = []
        for i, option_text_template in enumerate(EXTRA_OPTIONS_LIST):
            key_label_part = option_text_template.split('.')[0].replace(' ', '_').replace('/', '_').lower()
            if len(key_label_part) > 30: key_label_part = key_label_part[:30]
            extra_option_key = f"extra_option_{i}_{key_label_part}"
            if kwargs.get(extra_option_key, False): selected_extra_options_prompts.append(option_text_template)
        
        pil_images = [Image.fromarray(np.clip(255. * img.cpu().numpy().squeeze(),0,255).astype(np.uint8)).convert("RGB") for img in image]
        all_captions = []

        for input_image_pil in pil_images:
            actual_prompt_str = _build_prompt_beta_shared(caption_type, caption_length, selected_extra_options_prompts, character_name, custom_prompt)
            print(f"{self.NODE_NAME}: Prompt: {actual_prompt_str}")
            convo = [{"role": "system", "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."}, {"role": "user", "content": actual_prompt_str.strip()}]
            convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            
            model_device = model.device # Use the actual device of the loaded model
            inputs_on_device = processor(text=[convo_string], images=[input_image_pil], return_tensors="pt").to(model_device)
            inputs_on_device['pixel_values'] = inputs_on_device['pixel_values'].to(model.dtype) # Ensure correct dtype for pixel_values
            
            try:
                with torch.cuda.amp.autocast(enabled=("cuda" in str(model_device).lower() and model.dtype != torch.float32)):
                    generate_ids = model.generate(**inputs_on_device, max_new_tokens=max_new_tokens, do_sample=(temperature > 0), temperature=temperature if temperature > 0 else None, top_p=top_p if temperature > 0 else None, use_cache=True)
            except Exception as e:
                print(f"{self.NODE_NAME}: Generation error: {e}")
                if "out of memory" in str(e).lower() and "cuda" in str(model_device).lower(): 
                    print(f"{self.NODE_NAME}: OOM error detected. Clearing model cache."); _free_model_memory_shared()
                return ([f"Error generating caption: {e}"],)
            input_token_len = inputs_on_device.input_ids.shape[1]
            generated_text_ids = generate_ids[:, input_token_len:]
            caption = processor.batch_decode(generated_text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            all_captions.append(caption.strip())
        
        if not cache_model:
            print(f"{self.NODE_NAME}: Not caching model, releasing from memory.")
            _free_model_memory_shared()
        return (all_captions,)

class ExtraOptionsNode_Beta:
    CATEGORY = 'TTP_Toolset'
    FUNCTION = "compile_extra_options"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extra_options_str",)
    OUTPUT_IS_LIST = (False,) 

    def __init__(self):
        self.NODE_NAME = 'ExtraOptionsNode_Beta'

    @classmethod
    def INPUT_TYPES(cls):
        extra_options_inputs = {}
        for i, option_text in enumerate(EXTRA_OPTIONS_LIST):
            label = option_text.split('.')[0].replace(' ', '_').replace('/', '_').lower()
            if len(label) > 30: label = label[:30]
            extra_options_inputs[f"extra_option_{i}_{label}"] = ("BOOLEAN", {"default": False, "label": option_text[:100]})
        inputs = {
            "required": {
                "enable_extra_options": ("BOOLEAN", {"default": True, "label": "Enable Extra Options"}),
                "character_name": ("STRING", {"default": "", "multiline": False, "label": "Person/Character Name (for {name})"}),
            },
        }
        inputs["required"].update(extra_options_inputs)
        return inputs

    def compile_extra_options(self, enable_extra_options, character_name, **kwargs):
        if not enable_extra_options:
            return ("",)
        
        compiled_options = []
        for i, option_text_template in enumerate(EXTRA_OPTIONS_LIST):
            key_label_part = option_text_template.split('.')[0].replace(' ', '_').replace('/', '_').lower()
            if len(key_label_part) > 30: key_label_part = key_label_part[:30]
            extra_option_key = f"extra_option_{i}_{key_label_part}"
            if kwargs.get(extra_option_key, False):
                if option_text_template == NAME_OPTION_PROMPT:
                    if character_name: # Only add if name is provided
                        compiled_options.append(option_text_template.format(name=character_name))
                else:
                    compiled_options.append(option_text_template)
        return (" ".join(compiled_options),)

class JoyCaptionBetaOne_Simple:
    CATEGORY = 'TTP_Toolset'
    FUNCTION = "caption_image_simple"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.NODE_NAME = 'JoyCaptionBetaOne_Simple'
    
    @classmethod
    def INPUT_TYPES(cls):
        caption_type_keys = list(CAPTION_TYPE_MAP_BETA.keys())
        caption_length_list = ["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)]
        quantization_mode_list = ['bf16', 'nf4', 'int8'] 
        
        gpu_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not gpu_devices:
            gpu_devices = ["cpu"]
        return {
            "required": {
                "image": ("IMAGE",),
                "caption_type": (caption_type_keys,),
                "caption_length": (caption_length_list,),
                "quantization_mode": (quantization_mode_list, {"default": 'bf16'}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "label": "Custom Prompt (Overrides caption type & extras)"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "device": (gpu_devices,),
                "cache_model": ("BOOLEAN", {"default": True, "label": "Cache Model in Memory"}),
            },
            "optional": {
                "extra_options_str": ("STRING", {"forceInput": True, "default": ""}),
                 "enable_liger_kernel_opt": ("BOOLEAN", {"default": True, "label": "Enable LIGER Kernel (CUDA only)"}), # Liger as optional for simple
            }
        }

    @torch.no_grad()
    def caption_image_simple(self, image: torch.Tensor, caption_type: str, caption_length: str,
                             quantization_mode: str, custom_prompt: str,
                             temperature: float, top_p: float, max_new_tokens: int,
                             device: str, cache_model: bool, 
                             extra_options_str:str="", enable_liger_kernel_opt:bool=True):
        enable_liger_kernel = enable_liger_kernel_opt if LIGER_KERNEL_AVAILABLE else False
        try:
            model, processor = _load_model_shared(MODEL_PATH_HF_DEFAULT, quantization_mode, device, enable_liger_kernel)
        except Exception as e:
            print(f"Error in {self.NODE_NAME}: {e}")
            return ([str(e)],) # Return error message as list of strings
        
        list_of_extra_prompts = [extra_options_str] if extra_options_str.strip() else []

        pil_images = [Image.fromarray(np.clip(255. * img.cpu().numpy().squeeze(),0,255).astype(np.uint8)).convert("RGB") for img in image]
        all_captions = []

        for input_image_pil in pil_images:
            # For the simple node, character_name is not a direct input. If name handling is desired via extra_options_str,
            # it must have been compiled into that string by ExtraOptionsNode_Beta.
            # So, we pass an empty string for character_name_val to _build_prompt_beta_shared.
            actual_prompt_str = _build_prompt_beta_shared(caption_type, caption_length, list_of_extra_prompts, "", custom_prompt)
            print(f"{self.NODE_NAME}: Prompt: {actual_prompt_str}")
            convo = [{"role": "system", "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."}, {"role": "user", "content": actual_prompt_str.strip()}]
            convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            
            model_device = model.device
            inputs_on_device = processor(text=[convo_string], images=[input_image_pil], return_tensors="pt").to(model_device)
            inputs_on_device['pixel_values'] = inputs_on_device['pixel_values'].to(model.dtype)
            try:
                with torch.cuda.amp.autocast(enabled=("cuda" in str(model_device).lower() and model.dtype != torch.float32)):
                    generate_ids = model.generate(**inputs_on_device, max_new_tokens=max_new_tokens, do_sample=(temperature > 0), temperature=temperature if temperature > 0 else None, top_p=top_p if temperature > 0 else None, use_cache=True)
            except Exception as e:
                print(f"{self.NODE_NAME}: Generation error: {e}")
                if "out of memory" in str(e).lower() and "cuda" in str(model_device).lower(): 
                    print(f"{self.NODE_NAME}: OOM error detected. Clearing model cache."); _free_model_memory_shared()
                return ([f"Error generating caption: {e}"],)
            input_token_len = inputs_on_device.input_ids.shape[1]
            generated_text_ids = generate_ids[:, input_token_len:]
            caption = processor.batch_decode(generated_text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            all_captions.append(caption.strip())
        
        if not cache_model:
            print(f"{self.NODE_NAME}: Not caching model, releasing from memory.")
            _free_model_memory_shared()
        return (all_captions,)

NODE_CLASS_MAPPINGS = {
    "JoyCaptionBetaOne_Full": JoyCaptionBetaOne_Full,
    "ExtraOptionsNode_Beta": ExtraOptionsNode_Beta,
    "JoyCaptionBetaOne_Simple": JoyCaptionBetaOne_Simple,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyCaptionBetaOne_Full": "TTP_JoyCaption_BetaOne_Full",
    "ExtraOptionsNode_Beta": "TTP_ExtraOptionsNode_Beta",
    "JoyCaptionBetaOne_Simple": "TTP_JoyCaption_BetaOne_Simple",
}
print("JoyCaptionBetaOne (JCBO.py) nodes (Full, Simple, ExtraOptions) loaded with refined quantization.") 