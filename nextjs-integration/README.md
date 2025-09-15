# Hand Detection with Next.js & Vercel AI SDK

This integration uses your HuggingFace-hosted hand detection model with Next.js and Vercel AI SDK.

## Setup

### 1. Install Dependencies

```bash
npm install ai openai
npm install @vercel/ai
```

### 2. Start Python Backend

The Python backend loads your model from HuggingFace:

```bash
# Install dependencies
pip install fastapi uvicorn ultralytics pillow

# Start API server
python3 backend/api.py
# Server runs on http://localhost:8000
```

### 3. Environment Variables

Create `.env.local` in your Next.js app:

```env
PYTHON_API_URL=http://localhost:8000
# For production, use your deployed API URL
# PYTHON_API_URL=https://your-api.herokuapp.com
```

### 4. Use in Your App

```tsx
import { HandDetector } from '@/components/hand-detector';

export default function Page() {
  return <HandDetector />;
}
```

## API Endpoints

### POST /api/detect-hand
Detect hand/arm/not_hand in uploaded image.

Request:
```javascript
const formData = new FormData();
formData.append('image', file);

const response = await fetch('/api/detect-hand', {
  method: 'POST',
  body: formData
});
```

Response:
```json
{
  "class": "hand",
  "confidence": 0.98,
  "probabilities": {
    "hand": 0.98,
    "arm": 0.01,
    "not_hand": 0.01
  },
  "isHand": true,
  "isArm": false
}
```

## Deployment Options

### Deploy Backend to Render/Railway

1. Create `requirements.txt`:
```txt
fastapi==0.104.1
uvicorn==0.24.0
ultralytics==8.0.0
pillow==10.0.0
```

2. Create `Procfile`:
```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

3. Deploy to Render/Railway and get your API URL

### Deploy Frontend to Vercel

```bash
vercel deploy
```

Update environment variable:
```
PYTHON_API_URL=https://your-backend.onrender.com
```

## Using with Vercel AI SDK Streaming

For real-time streaming responses with gesture interpretation:

```tsx
import { useChat } from 'ai/react';

function GestureInterpreter() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: '/api/gesture-chat',
    initialMessages: [
      {
        id: '1',
        role: 'system',
        content: 'I help interpret hand gestures and signs.'
      }
    ]
  });

  // After detection, ask AI about the gesture
  const interpretGesture = async (detectionResult) => {
    if (detectionResult.isHand) {
      await handleSubmit({
        content: `What gesture might this be with ${detectionResult.confidence}% confidence?`
      });
    }
  };

  return (
    // Your UI here
  );
}
```

## Example: Real-time Webcam Detection

```tsx
'use client';

import { useEffect, useRef, useState } from 'react';

export function RealtimeDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const detectFrame = async () => {
      if (!videoRef.current) return;

      // Capture frame
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');

      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);

        canvas.toBlob(async (blob) => {
          if (!blob) return;

          const formData = new FormData();
          formData.append('image', blob);

          try {
            const response = await fetch('/api/detect-hand', {
              method: 'POST',
              body: formData
            });

            const result = await response.json();
            setDetections(prev => [...prev.slice(-9), result]);
          } catch (error) {
            console.error('Detection error:', error);
          }
        }, 'image/jpeg', 0.8);
      }
    };

    // Start webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Detect every 500ms
          intervalId = setInterval(detectFrame, 500);
        }
      });

    return () => {
      clearInterval(intervalId);
      // Stop webcam
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline />
      <div>
        Latest: {detections[detections.length - 1]?.class || 'Waiting...'}
      </div>
    </div>
  );
}
```

## Direct Model Usage (without backend)

If you want to use the model directly in Python:

```python
from ultralytics import YOLO

# Load directly from HuggingFace
model = YOLO('https://huggingface.co/EtanHey/hand-detection-3class/resolve/main/model.pt')

# Detect
results = model.predict('image.jpg')
probs = results[0].probs

# Check what was detected
if probs.top1 == 0:
    print(f"Hand detected: {probs.top1conf:.1%}")
elif probs.top1 == 1:
    print(f"Arm detected: {probs.top1conf:.1%}")
else:
    print("No hand/arm detected")
```

## Model Information

- **Model URL**: https://huggingface.co/EtanHey/hand-detection-3class
- **Architecture**: YOLOv8s-cls
- **Classes**: hand (0), arm (1), not_hand (2)
- **Accuracy**: >96%
- **Size**: ~3MB

## Troubleshooting

### CORS Issues
Make sure your backend allows your frontend origin:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-app.vercel.app"]
)
```

### Model Loading Slow
The model is cached after first download. First request may take longer.

### Detection Not Working
- Check image is RGB format
- Ensure image size is reasonable (model expects 224x224 but handles resizing)
- Verify backend is running and accessible