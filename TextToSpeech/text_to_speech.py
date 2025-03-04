import os 
import sys 
sys.path.append(os.path.basename(""))
sys.path.append("TextToSpeech")

from utils.s3Utils import S3Helper
from utils.logger import get_logger
from typing import Optional, Tuple, Callable

import soundfile as sf
from pydub import AudioSegment
from ZonosInference import ZonosInference

aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")



class TextToSpeech:
    def __init__(self, model_choice : str = "Zyphra/Zonos-v0.1-hybrid"):
        self.logger = get_logger("TextToSpeechHandler")
        self.s3Helper = S3Helper(aws_access_key, aws_secret_key, aws_region)
        self.zonos = ZonosInference(model_choice)
    
    def _save_tts_output(self, audio_array, sample_rate, file_name):
        """
        Write audio array to file
        """
        try:
            sf.write(file_name, audio_array, sample_rate)
            assert os.path.isfile(file_name)
        except Exception as e:
            self.logger.exception(f"Error writing audio file to : {file_name}")
            raise 
    
    def _get_output_filename(self, task_id):
        return f"/tmp/{task_id}.wav"

    def _convert_to_mp3(self, file_path: str):
        """
        Convert file to mp3
        """
        file_name = os.path.splitext(file_path)[0]
        mp3_file_path = f"{file_name}.mp3"

        try:
            audio = AudioSegment.from_file(file_path)
            audio.export(mp3_file_path, format="mp3")
            return mp3_file_path
        except Exception as e:
            self.logger.exception(e)
            ## do not raise the exception, send wav file back 
            ## if conversion fails
    
    def _get_local_filepath_speaker_audio(self, speaker_audio_path : str):
        audio_path = speaker_audio_path.split("/")[-1]
        return f"/tmp/{audio_path}"
    
    def _download_and_validate_speaker_audio_path(self, speaker_audio_path : str, speaker_audio_path_local):
        try:
            self.s3Helper.download_file("lalals", speaker_audio_path, speaker_audio_path_local)
            assert os.path.isfile(speaker_audio_path_local)
            return speaker_audio_path_local
        except Exception as e:
            self.logger.exception(f"Error downloading speaker audio : {e}")
            self.logger.error(f"Continuing without it!!!")
            return None

    def _get_local_filepath_prefix_audio(self, prefix_audio_path : str):
        audio_path = prefix_audio_path.split("/")[-1]
        return f"/tmp/{audio_path}"
    
    def _download_and_validate_prefix_audio_path(self, prefix_audio_path : str, prefix_audio_path_local):
        try:
            self.s3Helper.download_file("lalals", prefix_audio_path, prefix_audio_path_local)
            assert os.path.isfile(prefix_audio_path_local)
            return prefix_audio_path_local
        except Exception as e:
            self.logger.exception(f"Error downloading prefix audio : {e}")
            self.logger.error(f"Continuing without it!!!")
            return None

    def _get_output_key_s3(self, task_id : str, format = "mp3"):
        return f"projects/{task_id}.{format}"


    def run(self, 
            task_id : str, 
            text : str, 
            language : str = "en-us", 
            speaker_audio_path: Optional[str] = None,
            prefix_audio_path: Optional[str] = None,
            # Conditioning parameters
            emotion: Tuple[float, ...] = (1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2),
            vq_score: float = 0.78,
            fmax: float = 24000.0,
            pitch_std: float = 45.0,
            speaking_rate: float = 15.0,
            dnsmos_ovrl: float = 4.0,
            speaker_noised: bool = False,
            # Generation parameters
            cfg_scale: float = 2.0,
            top_p: float = 0.0,
            top_k: int = 0,
            min_p: float = 0.0,
            linear: float = 0.5,
            confidence: float = 0.4,
            quadratic: float = 0.0,
            seed: int = 420,
            randomize_seed: bool = True,
            unconditional_keys: Tuple[str, ...] = ("emotion",),
            max_new_tokens: int = 86 * 30,  # ~30 seconds
            progress_callback: Optional[Callable[[int, int], None]] = None,
):
        try:
            speaker_audio_path_local, prefix_audio_path_local = None, None
            if speaker_audio_path:
                speaker_audio_path_local = self._get_local_filepath_speaker_audio(speaker_audio_path)
                speaker_audio_path_local = self._download_and_validate_speaker_audio_path(speaker_audio_path, speaker_audio_path_local)

            if prefix_audio_path:
                prefix_audio_path_local = self._get_local_filepath_prefix_audio(prefix_audio_path)
                prefix_audio_path_local = self._download_and_validate_prefix_audio_path(prefix_audio_path, prefix_audio_path_local)


            sr, audio = self.zonos.run(text, language, speaker_audio_path_local, prefix_audio_path_local, 
                                       emotion, vq_score, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, 
                                       cfg_scale, top_p, top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, 
                                       unconditional_keys, max_new_tokens, progress_callback)
            
            output_filepath = self._get_output_filename(task_id)
            self._save_tts_output(audio, sr, output_filepath)
            output_filepath_mp3 = self._convert_to_mp3(output_filepath)
            s3_key = self._get_output_key_s3(task_id, format = "mp3")
            s3_key_wav = self._get_output_key_s3(task_id, format = "wav")
            self.s3Helper.upload_file(output_filepath, s3_key_wav, "lalals")
            self.s3Helper.upload_file(output_filepath_mp3, s3_key, "lalals")
            duration = AudioSegment.from_file(output_filepath).duration_seconds

            return {'success' : True, 'conversion_path' : s3_key, 'conversion_path_wav' : s3_key_wav, 'conversion_duration' : duration}          
        except Exception as e:
            self.logger.exception(e)
            return {'success' : False, 'error' : str(e)}