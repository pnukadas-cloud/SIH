"""
MAITRI — Offline Text-to-Speech (TTS) Engine
Synthesizes speech locally for space cabin audio output using pyttsx3.
"""

import threading
from typing import Optional

class TTSEngine:
    def __init__(self):
        self.enabled = True
        self._lock = threading.Lock()
        
    def speak_async(self, text: str, rate: int = 160, volume: float = 1.0):
        """Speak text in a background thread to prevent blocking real-time telemetry."""
        if not self.enabled or not text:
            return
            
        def _worker():
            with self._lock:
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate', rate)
                    engine.setProperty('volume', volume)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    # In headless or browser-dominant mode, browser Web Speech handles audio
                    pass
                    
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
