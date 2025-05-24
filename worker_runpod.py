import os, json, requests, random, time, runpod, base64
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np
import io

from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_flux, nodes_model_advanced, nodes_custom_sampler

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPVisionLoader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
StyleModelLoader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()

ModelSamplingFlux = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
StyleModelApply = NODE_CLASS_MAPPINGS["StyleModelApply"]()
CLIPVisionEncode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("flux1-dev.sft", "default")[0]
    clip = DualCLIPLoader.load_clip("t5xxl_fp16.safetensors", "clip_l.safetensors", "flux")[0]
    clip_vision = CLIPVisionLoader.load_clip("clip_vision.safetensors")[0]
    style_model = StyleModelLoader.load_style_model("flux1-redux-dev.safetensors")[0]
    vae = VAELoader.load_vae("ae.sft")[0]

def base64_to_image(base64_string):
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Create a temporary file to save the image
        temp_dir = '/content/ComfyUI/input'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f'temp_{time.time()}.png')
        
        # Save the bytes as an image file
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
            
        return temp_path
    except Exception as e:
        raise ValueError(f"Failed to process base64 image: {str(e)}")

@torch.inference_mode()
def generate(input):
    values = input["input"]
    temp_files = []

    try:
        # Convert base64 inputs to temporary image files
        input_image1 = base64_to_image(values['input_image1'])
        input_image2 = base64_to_image(values['input_image2'])
        temp_files.extend([input_image1, input_image2])

        positive_prompt = values['positive_prompt']
        seed = values['seed']
        steps = values['steps']
        guidance = values['guidance']
        sampler_name = values['sampler_name']
        scheduler = values['scheduler']
        max_shift = values['max_shift']
        base_shift = values['base_shift']
        width = values['width']
        height = values['height']

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        print(seed)

        image1 = LoadImage.load_image(input_image1)[0]
        image2 = LoadImage.load_image(input_image2)[0]
        conditioning_positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        conditioning_positive = FluxGuidance.append(conditioning_positive, guidance)[0]
        
        # Get crop method from input, default to "center"
        crop_method = values.get('crop_method', 'center')
        
        clip_vision_conditioning1 = CLIPVisionEncode.encode(clip_vision, image1, crop_method)[0]
        style_vision_conditioning1 = StyleModelApply.apply_stylemodel(clip_vision_conditioning1, style_model, conditioning_positive)[0]
        clip_vision_conditioning2 = CLIPVisionEncode.encode(clip_vision, image2, crop_method)[0]
        style_vision_conditioning2 = StyleModelApply.apply_stylemodel(clip_vision_conditioning2, style_model, style_vision_conditioning1)[0]
        unet_flux = ModelSamplingFlux.patch(unet, max_shift, base_shift, width, height)[0]
        noise = RandomNoise.get_noise(seed)[0]
        guider = BasicGuider.get_guider(unet_flux, style_vision_conditioning2)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(unet_flux, scheduler, steps, 1.0)[0]
        latent_image = EmptyLatentImage.generate(width, height)[0]
        samples, _ = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
        decoded = VAEDecode.decode(vae, samples)[0].detach()
        
        # Convert the image to base64
        img = Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0])
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "status": "DONE",
            "output": {
                "image": img_base64
            }
        }

    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e)
        }
    
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

runpod.serverless.start({"handler": generate}) 
