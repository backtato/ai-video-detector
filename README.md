# 🎥 AI-Video Detector — Backend (FastAPI)

**Versione:** 1.2.x  
**Autore:** Backtato / Adriano Brutti  
**Licenza:** MIT  
**Deploy compatibile:** [Render](https://render.com), Docker, o qualsiasi hosting Python 3.11+

---

## 🚀 Descrizione

**AI-Video Detector** è un backend scritto in **FastAPI** che analizza video e audio per stimare, con logica euristica e parametri conservativi, se un contenuto è **reale**, **generato da AI** o **incerto**.  
L’analisi combina segnali **video (frame, movimento, artefatti)** e **audio (voce, ritmo, musica)**, restituendo un punteggio normalizzato e un motivo sintetico del verdetto.

Questo backend è pensato per essere usato con il **plugin WordPress “AI-Video Detector”** (shortcode `[ai_video_checker]` o `[ai-video]`) ma può anche funzionare da API standalone.

---

## 📂 Struttura progetto

ai-video-detector/
├── api.py # App FastAPI principale
├── gunicorn_conf.py # Config Gunicorn (timeout, workers, ecc.)
├── requirements.txt # Librerie necessarie
├── Dockerfile # Build e deploy
├── render.yaml # Config per Render
├── .dockerignore
├── README.md # (questo file)
└── app/
├── analyzers/
│ ├── audio.py # Analisi audio (RMS, voce, musica, ecc.)
│ ├── video.py # Analisi video (frame, motion, artefatti)
│ ├── fusion.py # Fusione audio+video e calcolo finale
│ ├── meta.py # Estrazione metadati e device (ffprobe)
│ ├── forensic.py # Placeholder analisi C2PA / forense
│ └── heuristics_v2.py # Shim compatibilità
└── init.py
