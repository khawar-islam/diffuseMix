import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from accelerate import Accelerator


class ModelHandler:
    def __init__(self, model_id, device):
        self.accelerator = Accelerator()
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
        ).to(device)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)

    def generate_images(self, prompt, img_path, num_images, guidance_scale):
        image = Image.open(img_path).convert('RGB').resize((256, 256))
        return self.pipeline(prompt, image=image, num_images_per_prompt=num_images, guidance_scale=guidance_scale).images
