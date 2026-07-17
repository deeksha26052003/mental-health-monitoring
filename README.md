# Mental Health Monitoring System

A comprehensive, multi-modal mental health monitoring system that combines computer vision, sensor-based data, clinically validated questionnaires, and a Retrieval-Augmented Generation (RAG) layer to assess mental well-being and generate grounded, evidence-based recommendations.

📄 **Published research**: The multi-modal classification methodology (CNN + XGBoost + RAG) and evaluation results from this project were published in a peer-reviewed IEEE conference paper.

---

## Features

### 1. Contactless Respiration Monitoring
Monitors respiration rate without physical contact using camera-based detection.
- OpenCV for video processing
- Detects micro-movements in video frames to estimate respiration rate

### 2. Body Temperature Detection
Measures body temperature using an infrared sensor (MLX90614) or thermal imaging.
- Non-contact infrared temperature sensing
- Interfaces with a microcontroller (Raspberry Pi / Arduino / ESP32) via SDA/SCL

### 3. Depression Assessment via Questionnaire
A self-assessment questionnaire based on clinically validated scales (e.g., PHQ-9).
- Calculates a depression risk score from user responses on mood, energy, and mental state

### 4. Facial Expression Detection
Analyzes facial expressions to detect signs of stress, anxiety, or depression.
- CNN-based emotion classification (happiness, sadness, anger, fear, etc.)
- Works on live video or static image input

### 5. Multi-Modal Risk Classification + Grounded RAG Recommendations
Combines the CNN-based facial expression classifier with an XGBoost model trained on structured behavioral/questionnaire data, fusing both signals into a single risk classification (87% accuracy). Recommendations are then generated through a RAG pipeline so that every suggestion is grounded in a real knowledge base — not fabricated by the underlying LLM.

---

## System Architecture

```
Data Collection
  ├── Video/image (facial expression, respiration)
  ├── Infrared sensor (body temperature)
  └── Questionnaire responses (PHQ-9-based)
        │
        ▼
Preprocessing & Feature Extraction
        │
        ▼
  ┌─────────────────────┬──────────────────────┐
  │ CNN (facial          │ XGBoost (structured   │
  │ expression signal)   │ behavioral features)  │
  └─────────────────────┴──────────────────────┘
        │                         │
        └───────────┬─────────────┘
                     ▼
          Fused Risk Classification (87%)
                     │
                     ▼
     Retrieval-Augmented Generation (RAG) Layer
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Knowledge base retrieval   LLM recommendation
  (embeddings + cosine sim)  (grounded in retrieved context)
```

### Key files
- **`facial_expression.py`** — CNN-based facial expression/emotion classifier
- **`respiration.py`** — contactless respiration rate estimation from video
- **`model.py`** — XGBoost model training on structured behavioral/questionnaire data
- **`augment_data.py`** — data augmentation utilities for the training set
- **`knowledge_base.py`** — builds and indexes the RAG knowledge base using OpenAI embeddings and cosine similarity search
- **`rag_recommender.py`** — retrieval and recommendation logic; embeds the query, retrieves top-k relevant knowledge base entries, and constructs a grounded prompt for the LLM
- **`predict.py`** — main prediction pipeline: runs classification, then calls the RAG recommender for a grounded output
- **`main_menu.py`** — application entry point / menu for running individual features

---

## Key Engineering Decisions

**Why RAG instead of direct LLM generation?**
An early version of the recommendation engine used free-form LLM generation. This produced a critical failure case: the model hallucinated a crisis hotline number that did not correspond to any real, verified source — a serious issue given the sensitive nature of this domain. The fix was to constrain every recommendation to content retrieved from the knowledge base, and tighten the system prompt to explicitly forbid presenting ungrounded resources as fact.

**Why cosine similarity over FAISS?**
The RAG layer originally used `faiss-cpu` for vector search, which caused reproducible segmentation faults on macOS. This was replaced with OpenAI embeddings + cosine-similarity retrieval, trading some performance at scale for reliability and simplicity given the knowledge base size.

**Ablation studies**
Ablation studies were run isolating the individual contribution of the CNN, XGBoost, and RAG components to the final classification accuracy, validating the fused multi-modal architecture over simpler single-modality baselines.

---

## Hardware Setup (Temperature Sensor)

Uses the **MLX90614 infrared temperature sensor** for non-contact body temperature detection.

**Requirements:**
- MLX90614 sensor
- Microcontroller/board: Raspberry Pi, Arduino, or ESP32
- Breadboard, jumper wires, pull-up resistors (if required)
- Power supply (3.3V or 5V depending on the board)

**Wiring:**
| Pin | Connection |
|-----|-----------|
| VIN | 3.3V or 5V |
| GND | Ground |
| SDA | Data pin on microcontroller |
| SCL | Clock pin on microcontroller |

---

## Tech Stack

- **Languages:** Python
- **ML/CV:** PyTorch/TensorFlow, OpenCV, XGBoost
- **RAG:** OpenAI embeddings, cosine similarity search
- **API/GUI:** Flask, Streamlit/Tkinter
- **Data:** pandas, NumPy
- **Storage:** SQLite/Firebase (for user data and results)

---

## Installation

```bash
git clone https://github.com/deeksha26052003/mental-health-monitoring.git
cd mental-health-monitoring
pip install -r requirements.txt
python main_menu.py
```

## Usage

1. Launch the application and select the desired feature from the menu.
2. **Respiration monitoring** — ensure good lighting and position yourself in front of the camera.
3. **Temperature detection** — connect the MLX90614 sensor via the wiring above.
4. **Questionnaire** — complete the PHQ-9-based assessment honestly for an accurate score.
5. **Facial expression detection** — allow camera access and hold a neutral position during capture.
6. View classification results and grounded, RAG-generated recommendations.

---

## Honesty & Limitations

This is a research prototype, not a clinically validated diagnostic tool:
- Classification outputs come from a CNN + XGBoost ensemble trained on an augmented dataset, not verified against clinical ground truth by licensed professionals.
- The RAG knowledge base is a curated set of documents assembled for this project, not a comprehensive or continuously updated clinical resource.
- This system is not a substitute for professional mental health care.

---

## Future Enhancements

- Real-time multi-user monitoring support
- Wearable device integration for improved sensor accuracy
- Expanded knowledge base coverage for the RAG recommendation layer

---

## Citation

If referencing this work, please cite the associated IEEE conference paper (details available on request).
