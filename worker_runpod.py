import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import  nodes_flux, nodes_model_advanced, nodes_custom_sampler

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPVisionLoader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
StyleModelLoader =  NODE_CLASS_MAPPINGS["StyleModelLoader"]()

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

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image1 = values['input_image1']
    input_image1 = download_file(url=input_image1, save_dir='/content/ComfyUI/input', file_name='input_image1')
    input_image2 = values['input_image2']
    input_image2 = download_file(url=input_image2, save_dir='/content/ComfyUI/input', file_name='input_image2')
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
    clip_vision_conditioning1 = CLIPVisionEncode.encode(clip_vision, image1)[0]
    style_vision_conditioning1 = StyleModelApply.apply_stylemodel(clip_vision_conditioning1, style_model, conditioning_positive)[0]
    clip_vision_conditioning2 = CLIPVisionEncode.encode(clip_vision, image2)[0]
    style_vision_conditioning2 = StyleModelApply.apply_stylemodel(clip_vision_conditioning2, style_model, style_vision_conditioning1)[0]
    unet_flux = ModelSamplingFlux.patch(unet, max_shift, base_shift, width, height)[0]
    noise = RandomNoise.get_noise(seed)[0]
    guider = BasicGuider.get_guider(unet_flux, style_vision_conditioning2)[0]
    sampler = KSamplerSelect.get_sampler(sampler_name)[0]
    sigmas = BasicScheduler.get_sigmas(unet_flux, scheduler, steps, 1.0)[0]
    latent_image = EmptyLatentImage.generate(width, height)[0]
    samples, _ = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(f"/content/flux.1-dev-redux-{seed}-tost.png")

    result = f"/content/flux.1-dev-redux-{seed}-tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image1):
            os.remove(input_image1)
        if os.path.exists(input_image2):
            os.remove(input_image2)

runpod.serverless.start({"handler": generate})
