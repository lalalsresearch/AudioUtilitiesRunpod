

import os
import sys 
sys.path.append(os.path.basename(""))
sys.path.append("TextToSpeech")
from utils.logger import get_logger
from utils.aws_utils import S3Helper

import torch
import torchaudio
from typing import Optional, Tuple, Callable
# from beam import env

# if env.is_remote():
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

from pydub import AudioSegment
import re
import soundfile as sf


import numpy as np 

class ZonosInference:
    def __init__(self, model_choice: str = "Zyphra/Zonos-v0.1-hybrid"):
        self.logger = get_logger("ZonosInference")
        self.model_type = None
        self.model = None
        self.speaker_embedding = None
        self.speaker_audio_path = None
        self.load_model(model_choice)
        self.split_text_into_sentences = os.getenv("SPLIT_TEXT_INTO_SENTENCES", "true").strip().lower() == "true"

    def load_model(self, model_choice: str):
        """Load or switch between different Zonos models"""
        if self.model_type != model_choice:
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
            
            self.logger.debug(f"Loading {model_choice} model...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = Zonos.from_pretrained(model_choice, device=device)
            self.model.requires_grad_(False).eval()
            self.model_type = model_choice
            self.logger.debug(f"{model_choice} model loaded successfully!")

    def split_text(self, text: str) -> list:
        """Split text into multiple sentences, ensuring no segment exceeds 40 words."""
        sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by sentence-ending punctuation
        chunks = []
        current_chunk = []
        word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) > 40:
                chunks.append(" ".join(current_chunk))
                current_chunk = words
                word_count = len(words)
            else:
                current_chunk.extend(words)
                word_count += len(words)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def run(
        self,
        text: str,
        language: str = "en-us",
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
    ) -> Tuple[int, np.ndarray]:
        """
        Generate audio from text with optional conditioning parameters
        
        Returns:
            tuple: (sample_rate, audio_array)
        """
        self.logger.debug(f"Running inference for text : {text}")
        # Validate inputs
        if language not in supported_language_codes:
            raise ValueError(f"Unsupported language code: {language}")
        
        if len(emotion) != 8:
            raise ValueError("Emotion must be a tuple of 8 values")

        # Seed handling
        if randomize_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Speaker embedding handling
        if speaker_audio_path and "speaker" not in unconditional_keys:
            if speaker_audio_path != self.speaker_audio_path:
                wav, sr = torchaudio.load(speaker_audio_path)
                self.speaker_embedding = self.model.make_speaker_embedding(wav, sr)
                self.speaker_embedding = self.speaker_embedding.to(device, dtype=torch.bfloat16)
                self.speaker_audio_path = speaker_audio_path

        # Audio prefix handling
        audio_prefix_codes = None
        if prefix_audio_path:
            wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
            wav_prefix = wav_prefix.mean(0, keepdim=True)
            wav_prefix = self.model.autoencoder.preprocess(wav_prefix, sr_prefix)
            wav_prefix = wav_prefix.to(device, dtype=torch.float32)
            audio_prefix_codes = self.model.autoencoder.encode(wav_prefix.unsqueeze(0))

        # Prepare conditioning tensors
        emotion_tensor = torch.tensor(emotion, device=device)
        vq_tensor = torch.tensor([vq_score] * 8, device=device).unsqueeze(0)

        if self.split_text_into_sentences:
            text_segments = self.split_text(text)
        else:
            text_segments = [text]
        
        combined_audio = AudioSegment.silent(duration=10)
        chunk_filepath = "/tmp/chunk.wav"

        for segment in text_segments:
            torch.manual_seed(seed)
            cond_dict = make_cond_dict(
                text=segment,
                language=language,
                speaker=self.speaker_embedding,
                emotion=emotion_tensor,
                vqscore_8=vq_tensor,
                fmax=fmax,
                pitch_std=pitch_std,
                speaking_rate=speaking_rate,
                dnsmos_ovrl=dnsmos_ovrl,
                speaker_noised=speaker_noised,
                device=device,
                unconditional_keys=unconditional_keys,
            )
            conditioning = self.model.prepare_conditioning(cond_dict)

            # Generation callback
            def _progress_wrapper(frame: torch.Tensor, step: int, total_steps: int) -> bool:
                if progress_callback:
                    progress_callback(step, total_steps)
                return True

            # Generate codes
            codes = self.model.generate(
                prefix_conditioning=conditioning,
                audio_prefix_codes=audio_prefix_codes,
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                batch_size=1,
                sampling_params={
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "linear": linear,
                    "conf": confidence,
                    "quad": quadratic
                },
                callback=_progress_wrapper,
            )

            # Decode to audio
            wav_out = self.model.autoencoder.decode(codes).cpu().detach()
            sr_out = self.model.autoencoder.sampling_rate
            
            if wav_out.dim() == 2 and wav_out.size(0) > 1:
                wav_out = wav_out[0:1, :]
            wav_out = wav_out.squeeze().numpy()
            sf.write(chunk_filepath, wav_out, samplerate=sr_out)
            audio_segment = AudioSegment.from_wav(chunk_filepath)
            combined_audio += audio_segment + AudioSegment.silent(duration=500)
        if os.path.isfile(chunk_filepath):
            os.remove(chunk_filepath)
        final_audio = combined_audio.get_array_of_samples()
        combined_audio.export("/tmp/combined_audio.wav", format="wav")
        return sr_out, final_audio, "/tmp/combined_audio.wav"

# Example usage
if __name__ == "__main__":
    synthesizer = ZonosInference()
    
    # Simple generation
    sr, audio = synthesizer.run(
        text="This is a test of the Zonos text-to-speech system",
        language="en-us"
    )
    
    # Generation with more parameters
    sr, audio = synthesizer.run(
        text="Another test with different parameters",
        language="de-de",
        speaker_audio_path="speaker_sample.wav",
        emotion=(0.1, 0.8, 0.2, 0.3, 0.9, 0.1, 0.4, 0.5),
        cfg_scale=2.5,
        speaking_rate=18.0,
        randomize_seed=True
    )