import os 
import sys 
sys.path.append(os.path.basename(""))

import runpod 
from utils.response_utils import success, error
from utils.logger import get_logger
from text_to_speech import TextToSpeech

class ZonosInferenceRunpod():
    def __init__(self):
        try:
            self.logger = get_logger("ZonosInferenceRunpod")
            self.logger.debug(f"Initializing zonos tts")
            self.zonos = TextToSpeech()
            self.logger.debug(f"Successfully initialized zonos tts")
        except Exception as e:
            self.logger.exception(e)
            raise 
    
    def handler(self, event):
        try:
            task_id = event['input']['task_id']
            text = event['input']['text']
            language = event['input'].get("language", "en-us")
            speaker_audio_path = event['input'].get("speaker_audio_path")
            prefix_audio_path = event['input'].get("prefix_audio_path")
            emotion = event['input'].get("emotion", (1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2))
            vq_score = event['input'].get('vq_score', 0.78)
            fmax = event['input'].get('fmax', 24000.0)
            pitch_std = event['input'].get('pitch_std', 45.0)
            speaking_rate = event['input'].get('speaking_rate', 15.0)
            dnsmos_ovrl = event['input'].get('dnsmos_ovrl', 4.0)
            speaker_noised = event['input'].get('denoise_speaker', False)
            cfg_scale = event['input'].get('cfg_scale', 2.0)
            top_p = event['input'].get('top_p', 0.0)
            top_k = event['input'].get('top_k', 0)
            min_p = event['input'].get("min_p", 0.0)
            linear = event['input'].get("linear", 0.5)
            confidence = event['input'].get("confidence", 0.4)
            quadratic = event['input'].get("quadratic", 0.0)
            seed = event['input'].get("seed", 420)
            randomize_seed = event['input'].get("randomize_seed", True)
            unconditional_keys = event['input'].get("unconditional_keys", ("emotion",))
            max_new_tokens = event['input'].get("max_new_tokens", 86*30)
            progress_callback = None
            output = self.zonos.run(task_id, text, language, speaker_audio_path, prefix_audio_path, 
                                    emotion, vq_score, fmax, pitch_std, speaking_rate, dnsmos_ovrl, 
                                    speaker_noised, cfg_scale, top_p, top_k, min_p, linear, confidence,
                                    quadratic, seed, randomize_seed, unconditional_keys, max_new_tokens, 
                                    progress_callback)
            if output.get("success"):
                return success(output)
            else:
                return error(output)
        except Exception as e:
            self.logger.exception(e)
            out_obj = {
                'task_id' : task_id, 
                'success' : False
            }
            return error(out_obj)

if __name__ == "__main__":
    pipeline = ZonosInferenceRunpod()
    runpod.serverless.start({
        "handler" : pipeline.handler
    })