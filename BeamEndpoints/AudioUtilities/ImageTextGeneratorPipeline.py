from beam import function, task_queue, QueueDepthAutoscaler, Image, Volume
import os 
import sys 
sys.path.append(os.path.basename(''))

from ImageGenerator.image_generator_flux import ImageGenerator
@function(
    retries=0,
    cpu = 12, 
    memory = "32Gi",
    gpu = "A100-40",
    image=Image(
            # base_image="nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04",
            base_image="nvidia/cuda:12.4.0-devel-ubuntu22.04",
            python_version="python3.10", 
            python_packages="ImageGenerator/requirements.txt"
            ),
    volumes=[
        Volume(mount_path="./flux-models", name="flux-models"),
    ],
    secrets=["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_REGION", "AWS_BUCKET_NAME"])
def image_text_generator(mode, prompt = "", text = ""):
    if mode == "prompt2image":
        runner = ImageGenerator()
        output = runner.run(prompt)
        return output
    else:
        return {}


if __name__ == "__main__":
    image_text_generator( mode = "prompt2image", prompt="image of a cat holding a sign that says hi")