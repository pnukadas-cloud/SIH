"""
MAITRI — Audio Preprocessing & Acoustic Feature Extraction
Module: Voice Activity Detection (VAD), Pitch (F0) Tracking, Energy RMS,
Spectral Centroid, Vocal Jitter & Shimmer Biomarkers.
"""

import numpy as np
from typing import Dict, Any, Optional

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.pitch_history = []
        self.energy_history = []
        
    def process_audio_chunk(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Process a raw 1D float32 audio chunk (normalized between -1.0 and 1.0) at 16kHz.
        """
        if audio_data is None or len(audio_data) < 256:
            return self._default_empty_response()
            
        # Ensure float32 1D numpy array
        signal = np.asarray(audio_data, dtype=np.float32)
        if signal.ndim > 1:
            signal = signal.mean(axis=1) # downmix to mono
            
        # 1. Root Mean Square (RMS) Energy & Decibels
        rms_energy = float(np.sqrt(np.mean(signal**2)))
        db_energy = float(20 * np.log10(max(1e-5, rms_energy)))
        
        # 2. Voice Activity Detection (VAD threshold)
        is_speech = bool(rms_energy > 0.015)
        
        # 3. Zero Crossing Rate (ZCR)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal))
        zcr = float(zero_crossings)
        
        # 4. Pitch (F0) Extraction via Normalized Autocorrelation (Speech range 60Hz - 450Hz)
        pitch_hz = 0.0
        pitch_confidence = 0.0
        
        if is_speech and len(signal) >= 512:
            min_lag = int(self.sample_rate / 450.0) # ~35 samples at 16kHz
            max_lag = int(self.sample_rate / 65.0)  # ~246 samples at 16kHz
            
            # Autocorrelation
            corr = np.correlate(signal, signal, mode='full')
            corr = corr[len(corr)//2:] # take second half
            
            if len(corr) > max_lag:
                search_region = corr[min_lag:max_lag]
                peak_lag = np.argmax(search_region) + min_lag
                
                if corr[0] > 1e-6:
                    autocorr_val = corr[peak_lag] / corr[0]
                    if autocorr_val > 0.35: # strong periodicity
                        pitch_hz = float(self.sample_rate / peak_lag)
                        pitch_confidence = float(min(1.0, autocorr_val))
                        self.pitch_history.append(pitch_hz)
                        if len(self.pitch_history) > 50:
                            self.pitch_history.pop(0)

        # 5. Spectral Features via FFT
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1.0 / self.sample_rate)
        
        fft_sum = np.sum(fft_vals)
        if fft_sum > 1e-6:
            # Spectral Centroid (Center of Mass of frequencies)
            spectral_centroid = float(np.sum(freqs * fft_vals) / fft_sum)
            # Spectral Rolloff (85% energy frequency)
            cum_energy = np.cumsum(fft_vals)
            rolloff_idx = np.where(cum_energy >= 0.85 * fft_sum)[0]
            spectral_rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0
        else:
            spectral_centroid = 0.0
            spectral_rolloff = 0.0
            
        # 6. Vocal Jitter (Pitch Stability Index) & Shimmer (Amplitude Stability)
        if len(self.pitch_history) >= 5:
            pitch_diffs = np.abs(np.diff(self.pitch_history))
            jitter_local = float(np.mean(pitch_diffs) / max(10.0, np.mean(self.pitch_history)))
        else:
            jitter_local = 0.02
            
        # 7. Vocal Tension / Tremor Indicator
        # High pitch variability + high jitter + elevated spectral centroid indicates acute panic or vocal strain
        vocal_tension_score = float(np.clip((jitter_local * 15.0) + (zcr * 1.5) + (max(0.0, pitch_hz - 220.0) / 300.0), 0.0, 1.0))
        
        return {
            "is_speech_active": is_speech,
            "rms_energy": round(rms_energy, 4),
            "db_level": round(db_energy, 1),
            "pitch_f0_hz": round(pitch_hz, 1),
            "pitch_confidence": round(pitch_confidence, 2),
            "zero_crossing_rate": round(zcr, 4),
            "spectral_centroid_hz": round(spectral_centroid, 1),
            "spectral_rolloff_hz": round(spectral_rolloff, 1),
            "vocal_jitter": round(jitter_local, 4),
            "vocal_tension_score": round(vocal_tension_score, 3)
        }
        
    def _default_empty_response(self) -> Dict[str, Any]:
        return {
            "is_speech_active": False,
            "rms_energy": 0.0,
            "db_level": -80.0,
            "pitch_f0_hz": 0.0,
            "pitch_confidence": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "vocal_jitter": 0.0,
            "vocal_tension_score": 0.0
        }
