from beam import function, task_queue, QueueDepthAutoscaler, Image, Volume
import os 
import sys 
sys.path.append(os.path.basename(''))
from ImageGenerator.SanaSprint.image_generator_sana import ImageGenerator
 
os.environ["HF_HOME"] = "./flux-models/"

@function(
    retries=0,
    cpu = 12, 
    memory = "32Gi",
    gpu = "RTX4090",
    image=Image(
            base_image="sakarlalals/image-generator-sana-sprint",
            python_version="python3.11", 
            # python_packages="TextToSpeech/requirements.txt", 
            # commands=[]
            ),
    
    volumes=[
        Volume(mount_path="./flux-models", name="flux-models"),
    ],
    secrets=["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_REGION", "AWS_BUCKET_NAME", "HF_TOKEN"])
def image_generator(task_id : str, prompt : str, height : int = 1024, width : int = 1024, seed : int = 0):
    image_generator = ImageGenerator()
    return image_generator.run(task_id, prompt, height, width, seed)



if __name__ == "__main__":
    output = image_generator("1234", "a dog sitting on a chair, with a cat sitting on its side")
    print(output)
