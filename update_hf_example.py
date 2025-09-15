#!/usr/bin/env python3
"""
Update HuggingFace model card with cleaner examples
"""

from huggingface_hub import HfApi

def update_model_card():
    api = HfApi()
    repo_id = "EtanHey/hand-detection-3class"

    model_card = """---
tags:
- yolov8
- image-classification
- hand-detection
- computer-vision
library_name: ultralytics
---

# Hand Detection Model (YOLOv8)

This model classifies images into three categories:
- **hand**: Close-up hand with fingers visible (✋)
- **arm**: Forearm or elbow area (💪)
- **not_hand**: Neither hand nor arm (❌)

## Quick Start

```python
from ultralytics import YOLO

# Load model directly from HuggingFace
model = YOLO('https://huggingface.co/EtanHey/hand-detection-3class/resolve/main/model.pt')

# Predict on an image
results = model.predict('image.jpg')

# Get the prediction
probs = results[0].probs
class_id = probs.top1  # 0=arm, 1=hand, 2=not_hand (alphabetical order!)
confidence = probs.top1conf.item()

# Interpret results
if class_id == 1:  # hand is index 1
    print(f"✋ Hand detected: {confidence:.1%}")
elif class_id == 0:  # arm is index 0
    print(f"💪 Arm detected: {confidence:.1%}")
else:  # not_hand is index 2
    print(f"❌ No hand/arm detected: {confidence:.1%}")
```

## Live Demo (Webcam)

```python
import cv2
from ultralytics import YOLO

model = YOLO('https://huggingface.co/EtanHey/hand-detection-3class/resolve/main/model.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    probs = results[0].probs

    # YOLO uses alphabetical order!
    classes = ['arm', 'hand', 'not_hand']  # 0=arm, 1=hand, 2=not_hand
    label = f"{classes[probs.top1]}: {probs.top1conf:.1%}"

    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Use with Vercel AI SDK

```bash
npm install ai openai
```

```typescript
// app/components/hand-detector.tsx
'use client';

import { useChat } from 'ai/react';
import { useState } from 'react';

export function HandDetectorWithAI() {
  const [detection, setDetection] = useState(null);
  const { messages, input, handleSubmit } = useChat({
    api: '/api/chat',
    initialMessages: [{
      role: 'system',
      content: 'You help interpret hand gestures and signs.'
    }]
  });

  const detectAndAnalyze = async (file) => {
    // 1. Detect hand
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/api/detect-hand', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();
    setDetection(result);

    // 2. If hand detected, ask AI about gesture
    if (result.class === 'hand') {
      await handleSubmit({
        preventDefault: () => {},
        currentTarget: {
          input: { value: `What gesture is this hand making? Confidence: ${result.confidence}%` }
        }
      });
    }
  };

  return (
    <div>
      <input type="file" onChange={(e) => detectAndAnalyze(e.target.files[0])} />
      {detection && <p>Detected: {detection.class} ({detection.confidence}%)</p>}
      {messages.map(m => (
        <div key={m.id}>{m.role}: {m.content}</div>
      ))}
    </div>
  );
}
```

```typescript
// app/api/chat/route.ts
import { OpenAIStream, StreamingTextResponse } from 'ai';

export async function POST(req: Request) {
  const { messages } = await req.json();

  // Your OpenAI/AI provider logic here
  const stream = OpenAIStream(response);
  return new StreamingTextResponse(stream);
}
```

## Use in Next.js/Node.js

### Option 1: FastAPI Backend + Next.js

**Backend (Python):**
```python
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO('https://huggingface.co/EtanHey/hand-detection-3class/resolve/main/model.pt')

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model.predict(image)
    probs = results[0].probs

    return {
        "class": ['arm', 'hand', 'not_hand'][probs.top1],  # alphabetical order
        "confidence": float(probs.top1conf)
    }
```

**Frontend (Next.js):**
```javascript
async function detectHand(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    console.log(`Detected: ${result.class} (${result.confidence * 100}%)`);
}
```

### Option 2: ONNX for Browser

```bash
# Convert to ONNX first
from ultralytics import YOLO
model = YOLO('model.pt')
model.export(format='onnx')
```

Then use with ONNX Runtime Web:
```javascript
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('/model.onnx');
// Process and run inference...
```

## React Native

```javascript
const detectHand = async (imageUri) => {
    const formData = new FormData();
    formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'photo.jpg'
    });

    const response = await fetch('YOUR_API_URL/detect', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    Alert.alert(`Detected: ${result.class}`);
};
```

## cURL Test

```bash
curl -X POST -F "file=@test.jpg" http://localhost:8000/detect
```

## Model Details

- **Architecture**: YOLOv8s-cls (5M parameters)
- **Classes**: 3 (arm=0, hand=1, not_hand=2) - alphabetical order
- **Input Size**: 224x224
- **Accuracy**: >96% on validation set
- **Size**: ~3MB

## Training Data

- **Total Images**: 1,740
- **Distribution**:
  - Hand: 704 images (40%)
  - Arm: 320 images (18%)
  - Not Hand: 462 images (27%)
  - Val: 254 images (15%)

## Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 96.3% |
| Inference Speed | 30+ FPS (Apple M1) |
| Model Size | 2.97 MB |

## License

MIT - Free for commercial use

## Citation

If you use this model, please cite:
```
@software{hand_detection_yolo_2024,
  author = {EtanHey},
  title = {Hand Detection YOLOv8 Model},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/EtanHey/hand-detection-3class}
}
```
"""

    try:
        # Upload updated model card
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model card with cleaner examples"
        )
        print("✅ Model card updated successfully!")
        print(f"View at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error updating: {e}")

if __name__ == "__main__":
    update_model_card()