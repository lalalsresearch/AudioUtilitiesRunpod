

import os
import sys 
sys.path.append(os.path.basename(""))

import torch 
from diffusers import SanaSprintPipeline

from utils.logger import get_logger
from utils.aws_utils import S3Helper

from beam import env
if env.is_remote():
    from huggingface_hub import login 
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    os.environ["HF_HOME"] = "./flux-models/"
    os.environ["HF_CACHE_DIR"] = "./flux-models/"


aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

class ImageGenerator:
    def __init__(self):
        self.logger = get_logger("ImageGenerator")
        self.s3Helper = S3Helper(aws_access_key, aws_secret_key, aws_region)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        self.logger.debug("Initiating Sana Sprint pipeline")
        self.pipeline = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16, 
            cache_dir = "./flux-models/"
        ).to(self.device)
        self.logger.debug(f"Successfully initiated sana sprint pipeline")

    def run(self, task_id : str, prompt : str, height : int = 1024, width : int = 1024, seed : int = 0):
        self.logger.debug(f"Starting image generation for prompt : {prompt}")
        image = self.pipeline(prompt, height=height, width=width, generator=torch.Generator(device=self.device).manual_seed(seed)
                            )
        out_image = image.images[0]
        out_image.save("/tmp/flux-image.png")

        self.s3Helper.upload_file("/tmp/flux-image.png", f"ImageGenerator/{task_id}.png")

        return {'success' : True, 'image_path' : f"ImageGenerator/{task_id}.png"}

