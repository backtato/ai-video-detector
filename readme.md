# 🧠 AI-Video Detector

**AI-Video Detector** è un sistema open-source per analizzare rapidamente se un video è stato **probabilmente generato da intelligenza artificiale (AI)**.  
Combina analisi di **metadati**, **artefatti visivi** e **tracce audio** per stimare la plausibilità che un contenuto sia sintetico, deepfake o rielaborato digitalmente.

---

## ⚙️ Funzionalità principali

- 🔍 **Analisi file** – Carica un video e ricevi un verdetto probabilistico.  
- 🌐 **Analisi URL** – Incolla un link (MP4, WebM, M3U8, ecc.) e ottieni il risultato.  
- 🔗 **Analisi contesto** – Per domini noti (es. YouTube) senza scaricare l’intero video.  
- 🧮 **Output JSON strutturato** – Include punteggio, confidenza e dettagli tecnici.  
- 🕊️ **Privacy by design** – I video non vengono mai salvati.  
- 🧩 **Integrazione WordPress** tramite plugin [AI-Video Detector WP](https://github.com/backtato/ai-video-detector-wp).  

---

## 🧩 Architettura del progetto
ai-video-detector/
├── app.py # Entry principale FastAPI
├── calibration.py # Combinazione e calibrazione punteggi
├── detectors/ # Moduli di analisi
│ ├── metadata.py # Analisi metadati con ffprobe
│ ├── frame_artifacts.py # Analisi artefatti visivi (OpenCV)
│ └── audio.py # Euristiche su codec e sample rate
├── resolver.py # Downloader sicuro con limiti e filtri
├── requirements.txt
├── Dockerfile
└── README.md # Questo file

---


---

## 🧠 Logica di analisi

| Componente | Metodo | Esempi di segnali |
|-------------|---------|------------------|
| **Metadata** | ffprobe | encoder mancante, assenza `creation_time`, FPS anomalo |
| **Frame artifacts** | OpenCV | bassa energia alta-frequenza, eccessiva blockiness, rumore uniforme |
| **Audio** | euristiche | codec non lineare, assenza traccia, sample rate irregolare |
| **Calibrazione** | funzione smoothstep (3x²−2x³) | riduce overconfidence e rende il punteggio più stabile |
| **Decisione** | pesi configurabili + soglie | `Probabile AI`, `Probabile autentico`, `Inconcludente` |

---

## 🖥️ Requisiti

- Python ≥ 3.11  
- ffmpeg installato nel sistema (necessario per metadati e HLS)  
- Librerie Python:
  - `fastapi`
  - `uvicorn`
  - `gunicorn`
  - `requests`
  - `numpy`
  - `opencv-python-headless`

---

## 💻 Installazione locale

```bash
# Clona il progetto
git clone https://github.com/backtato/ai-video-detector.git
cd ai-video-detector

# Crea ambiente virtuale
python -m venv .venv && source .venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt

# Avvia il server FastAPI
uvicorn app:app --reload --port 8000
