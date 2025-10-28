# app/analyzers/__init__.py

# Esporta esplicitamente i moduli interni del pacchetto analyzers.
# Questo permette import come: from app.analyzers import video, audio, fusion

from . import video
from . import audio
from . import fusion

__all__ = ["video", "audio", "fusion"]
