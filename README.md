# Introduction:

Wrapped Joy Caption alpha 2 node for comfyui from https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two
Easy use, for GPU with less 19G, please use nf4 for better balanced speed and result.
This Node also took a reference from /chflame163/ComfyUI_LayerStyle and https://huggingface.co/John6666/joy-caption-alpha-two-cli-mod

# Usage:
-inside comfyui

-click comfyui manager

-Install Via git Url https://github.com/TTPlanetPig/Comfyui_JC2 or inside the folder ./comfyui/custom_nodes run 
```shell
git clone https://github.com/TTPlanetPig/Comfyui_JC2
```
-for python_embeded comfyui version, inside folder ./comfyui/custom_nodes/Comfyui_JC2 run 
```shell
../../../python_embeded/python.exe -m pip install -r requirements.txt
```

I will assume you have the pytorch ready in your PC, in case you don't, install from here: 
```shell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121  
# (NOT Recommended if you don't familar with embeded python)
```
# Model preparation:

-clip vision: https://huggingface.co/google/siglip-so400m-patch14-384/tree/main download all files and place in ComfyUI\models\clip_vision\google--siglip-so400m-patch14-384

-LLM: https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct download all files and place in ComfyUI\models\LLM\unsloth--Meta-Llama-3.1-8B-Instruct

-Joy capiton lora: https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two download all files and place in ComfyUI\models\Joy_caption\cgrkzexw-599808, i will suggest you use huggingface-cli to avoid mistaken on the names.

* Or you can download the models here and unzip it into your folder *
* https://pan.baidu.com/s/1yYRlDKclehSPv-tUVwfVHw?pwd=b84c *

Make sure your model is well placed as below.

![image](https://github.com/user-attachments/assets/510d2e6b-db1f-4743-92f4-9a8ae80ef6dd)


![image](https://github.com/user-attachments/assets/9ae0a410-539e-49c5-a1b4-4434da02dc28)

![image](https://github.com/user-attachments/assets/2d17e8d2-42af-4040-9cf9-019eb25464e0)

![image](https://github.com/user-attachments/assets/aeba0145-81c7-4c86-a31c-bbb9c317cad8)



# Key options:
1. For 3090/4090 use bf16 otherwise nf4. please notice nf4 accuracy is lower than bf16, you can directly figure out.
   ![image](https://github.com/user-attachments/assets/8001e70b-cea3-4971-a8c2-f483a2c4f91c) 
3. For differnt purpose on the prompts
   
   ![image](https://github.com/user-attachments/assets/110f25f6-ea25-4395-b698-c0ec358940ae)
5. not always longer is better:
   
   ![image](https://github.com/user-attachments/assets/05e8cfbe-f983-4c8e-813a-761779d0ba4e)
7. if enabled, the loaded clip,llm,lora model will not offload
   
   ![image](https://github.com/user-attachments/assets/804d3326-0f44-4cd2-98c9-56e174e552c1)
9. if enabled, all the extra option will be effective. if disabled, even you enabled in detail options.
    
    ![image](https://github.com/user-attachments/assets/6cb00a63-a1e6-4502-87ff-b99800d37912)
11. Must use together to be affective,
    
    ![image](https://github.com/user-attachments/assets/16d11016-6ff1-4d62-90ca-c3d820af4cd3),
    ![image](https://github.com/user-attachments/assets/6fe8dbd4-affe-4753-b10e-aa4120ab5149)



Enjoy!

## Star History

<a href="https://star-history.com/#TTPlanetPig/Comfyui_JC2&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_JC2&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_JC2&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_JC2&type=Date" />
 </picture>
</a>



