import os, json, random, time, runpod
from io import BytesIO
from PIL import Image
import base64
import numpy as np

import torch
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

# 8의 배수로 조정하는 헬퍼
def adjust_to_multiple_of_8(value):
    return max((value // 8) * 8, 8)

# 전역 모델 로딩
print("Loading models...")
with torch.inference_mode():
    unet = UNETLoader.load_unet("flux1-dev.sft", "default")[0]
    clip = DualCLIPLoader.load_clip("t5xxl_fp16.safetensors", "clip_l.safetensors", "flux")[0]
    clip_vision = CLIPVisionLoader.load_clip("clip_vision.safetensors")[0]
    style_model = StyleModelLoader.load_style_model("flux1-redux-dev.safetensors")[0]
    vae = VAELoader.load_vae("ae.sft")[0]
print("Models loaded successfully!")

def base64_to_temp_image(b64_string, temp_path):
    """Base64 문자열을 임시 이미지 파일로 저장"""
    # data:image/... 부분이 있다면 제거
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[-1]
    
    image_bytes = base64.b64decode(b64_string)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image.save(temp_path)
    return image

@torch.inference_mode()
def generate(input):
    try:
        values = input["input"]

        # Base64 이미지 입력
        input_image1_b64 = values['input_image1']
        input_image2_b64 = values['input_image2']
        positive_prompt = values['positive_prompt']
        seed = values.get('seed', 0)
        steps = values.get('steps', 25)
        guidance = values.get('guidance', 3.5)
        sampler_name = values.get('sampler_name', "euler")
        scheduler = values.get('scheduler', "normal")
        max_shift = values.get('max_shift', 1.15)
        base_shift = values.get('base_shift', 0.5)
        width = values.get('width', 1024)
        height = values.get('height', 1024)

        # 8의 배수로 크기 조정
        width = adjust_to_multiple_of_8(width)
        height = adjust_to_multiple_of_8(height)

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        print(f"Using seed: {seed}")

        # 임시 파일 경로
        temp_image1_path = f"/tmp/input_image1_{seed}.png"
        temp_image2_path = f"/tmp/input_image2_{seed}.png"

        # Base64를 임시 이미지 파일로 변환
        base64_to_temp_image(input_image1_b64, temp_image1_path)
        base64_to_temp_image(input_image2_b64, temp_image2_path)

        # 이미지 로딩 및 처리
        image1 = LoadImage.load_image(temp_image1_path)[0]
        image2 = LoadImage.load_image(temp_image2_path)[0]
        
        conditioning_positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        conditioning_positive = FluxGuidance.append(conditioning_positive, guidance)[0]
        
        clip_vision_conditioning1 = CLIPVisionEncode.encode(clip_vision, image1)[0]
        style_vision_conditioning1 = StyleModelApply.apply_stylemodel(
            clip_vision_conditioning1, style_model, conditioning_positive)[0]
        
        clip_vision_conditioning2 = CLIPVisionEncode.encode(clip_vision, image2)[0]
        style_vision_conditioning2 = StyleModelApply.apply_stylemodel(
            clip_vision_conditioning2, style_model, style_vision_conditioning1)[0]
        
        unet_flux = ModelSamplingFlux.patch(unet, max_shift, base_shift, width, height)[0]
        noise = RandomNoise.get_noise(seed)[0]
        guider = BasicGuider.get_guider(unet_flux, style_vision_conditioning2)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(unet_flux, scheduler, steps, 1.0)[0]
        latent_image = EmptyLatentImage.generate(width, height)[0]
        
        samples, _ = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
        decoded = VAEDecode.decode(vae, samples)[0].detach()
        
        # PIL Image로 변환
        output_array = np.array(decoded * 255, dtype=np.uint8)[0]
        output_image = Image.fromarray(output_array)
        
        # Base64로 인코딩
        buffer = BytesIO()
        output_image.save(buffer, format='PNG')
        output_b64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            'status': 'success',
            'output': f'data:image/png;base64,{output_b64}',
            'seed': seed,
            'width': width,
            'height': height
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }
    
    finally:
        # 임시 파일 정리
        for temp_file in [temp_image1_path, temp_image2_path]:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

runpod.serverless.start({"handler": generate})
