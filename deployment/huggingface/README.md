---
title: Hand Detection API
emoji: 🤚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
models:
  - EtanHey/hand-sign-detection
---

# Hand/Arm Detection API

This Space provides both a Gradio UI and API endpoints for hand/arm detection.

## Features
- 96.3% accuracy
- Real-time detection
- API endpoints for programmatic access
- Three classes: hand, arm, not_hand

## API Usage

### REST API Endpoint
```python
import requests

# File upload
response = requests.post(
    "https://etanhey-hand-detection-api.hf.space/api/predict",
    files={"file": open("image.jpg", "rb")}
)
print(response.json())

# Base64 image
import base64

with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://etanhey-hand-detection-api.hf.space/api/predict/base64",
    json={"image": image_base64}
)
print(response.json())
```

## Response Format
```json
{
  "class": "hand",
  "confidence": 0.963,
  "probabilities": {
    "hand": 0.963,
    "arm": 0.025,
    "not_hand": 0.012
  }
}
```

## Model
View the model: [EtanHey/hand-sign-detection](https://huggingface.co/EtanHey/hand-sign-detection)
