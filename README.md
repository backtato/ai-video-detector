## Aurora-Noor Detector — Backend (FastAPI)

**Versione:** 1.2.x  
**Autore:** Backtato / Adriano Brutti  
**Licenza:** MIT  
**Deploy compatibile:** [Render](https://render.com), Docker, o qualsiasi hosting Python 3.11+

---

##  Descrizione

**AI-Video Detector** è un backend scritto in **FastAPI** che analizza video e audio per stimare, con logica euristica e parametri conservativi, se un contenuto è **reale**, **generato da AI** o **incerto**.  
L’analisi combina segnali **video (frame, movimento, artefatti)** e **audio (voce, ritmo, musica)**, restituendo un punteggio normalizzato e un motivo sintetico del verdetto.

Questo backend è pensato per essere usato con il **plugin WordPress “AI-Video Detector”** (shortcode `[ai_video_checker]` ma può anche funzionare da API standalone.

