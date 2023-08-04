from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import WhisperModel
from faster_whisper.utils import download_model, format_timestamp
from faster_whisper.version import __version__

# Public interface of the faster_whisper package
__all__ = [
    "decode_audio",  # Module for audio decoding
    "WhisperModel",  # Module for the WhisperModel class
    "download_model",  # Function for downloading models
    "format_timestamp",  # Function for formatting timestamps
    "__version__",  # Package version
]
