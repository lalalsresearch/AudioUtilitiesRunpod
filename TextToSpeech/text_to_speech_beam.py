from beam import function, task_queue, QueueDepthAutoscaler, Image, Volume
import os 
import sys 
sys.path.append(os.path.basename(''))
from TextToSpeech.text_to_speech import TextToSpeech

custom_cache_dir = "./flux-models"
os.environ["HF_CACHE_DIR"] = "./flux-models"
os.environ["HF_HOME"] = "./flux-models"


@function(
    retries=0,
    cpu = 12, 
    memory = "32Gi",
    gpu = "RTX4090",
    image=Image(
            base_image="sakarlalals/text-to-speech-zonos",
            python_version="python3.11", 
            # python_packages="TextToSpeech/requirements.txt", 
            # commands=[]
            ),
    
    volumes=[
        Volume(mount_path="./flux-models", name="flux-models"),
    ],
    secrets=["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_REGION", "AWS_BUCKET_NAME"])
def text_to_speech(task_id : str, text : str):
    text_to_speech = TextToSpeech()
    return text_to_speech.run(task_id, text)



if __name__ == "__main__":
    output = text_to_speech("1234", "hello this is sakar trying out the new zonos tts model. I don't know why this is not working though.")
    print(output)