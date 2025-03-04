

import os
import sys 
sys.path.append(os.path.basename(""))

import torch 
from diffusers import FluxPipeline

from utils.logger import get_logger
from utils.aws_utils import S3Helper

from huggingface_hub import login 
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

custom_cache_dir = "./flux-models/"
os.environ["HF_CACHE_DIR"] = "./flux-models"
os.environ["HF_HOME"] = "./flux-models"

custom_cache_dir = "/runpod-volume/flux"
os.environ["HF_CACHE_DIR"] = "/runpod-volume/flux"
os.environ["HF_HOME"] = "/runpod-volume/flux"

aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

class ImageGenerator:
    def __init__(self):
        self.logger = get_logger("ImageTextGenerator")
        self.s3Helper = S3Helper(aws_access_key, aws_secret_key, aws_region)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        self.logger.debug("Initiating flux pipeline")
        self.fluxPipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=self.dtype, 
            cache_dir = custom_cache_dir
        ).to(self.device)
        self.logger.debug(f"Successfully initiated flux pipeline")

    def run(self, task_id : str, prompt : str, height : int = 1024, width : int = 1024, seed : int = 0):
        self.logger.debug(f"Starting image generation for prompt : {prompt}")
        image = self.fluxPipeline(prompt, 
                                  height = height, 
                                  width = width, 
                                  guidance_scale = 3.5, 
                                  max_sequence_length = 512, 
                                  generator = torch.Generator(self.device).manual_seed(seed))
        out_image = image.images[0]
        out_image.save("/tmp/flux-image.png")

        self.s3Helper.upload_file("/tmp/flux-image.png", f"FluxImages/{task_id}.png")

        return {'success' : True, 'image_path' : f"FluxImages/{task_id}.png"}

