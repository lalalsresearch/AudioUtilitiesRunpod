import os 
import sys 
sys.path.append(os.path.basename(""))

import runpod 
from utils.s3Utils import S3Helper
from utils.response_utils import success, error
from utils.logger import get_logger
from image_generator import ImageGenerator


class ImageGeneratorRunpod():
    def __init__(self):
        try:
            self.logger = get_logger("Image Generator")
            self.logger.debug(f"Initializing image generator")
            self.generator = ImageGenerator()
            self.logger.debug(f"successfully initialized image generator")
        except Exception as e:
            self.logger.error("Error initializing image generator")
            raise 
    
    def handler(self, event):
        try:
            task_id = event['input']['arguments']['task_id']
            prompt = event['input']['arguments']['prompt']
            height = event['input']['arguments'].get('height', 1024)
            width = event['input']['arguments'].get('width', 1024)
            seed = event['input']['arguments'].get('seed', 0)
            output = self.generator.run(task_id, prompt, height, width, seed)
            return success(output)
        except Exception as e:
            self.logger.exception(e)
            return error({'error' : str(e)})

if __name__ == "__main__":
    pipeline = ImageGeneratorRunpod()
    runpod.serverless.start({
        "handler" : pipeline.handler
    })