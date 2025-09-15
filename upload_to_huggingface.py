#!/usr/bin/env python3
"""
Upload model to HuggingFace Hub for direct URL access
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import argparse

def upload_model_to_hf(
    model_path="models/three_class_detector.pt",
    repo_name="hand-detection-yolo",
    username=None,  # Will use logged-in user if None
    private=False
):
    """
    Upload YOLO model to HuggingFace Hub

    Usage:
        python3 upload_to_huggingface.py

    Then use in your app:
        from ultralytics import YOLO
        model = YOLO('https://huggingface.co/YOUR_USERNAME/hand-detection-yolo/resolve/main/model.pt')
    """

    # Initialize API
    api = HfApi()

    # Get username if not provided
    if username is None:
        try:
            user_info = api.whoami()
            username = user_info["name"]
            print(f"✅ Logged in as: {username}")
        except:
            print("❌ Not logged in to HuggingFace!")
            print("Run: huggingface-cli login")
            return None

    # Full repository ID
    repo_id = f"{username}/{repo_name}"

    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository ready: {repo_id}")
    except Exception as e:
        print(f"Repository exists or error: {e}")

    # Upload model file
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Model not found: {model_path}")
        return None

    print(f"📤 Uploading {model_file.name}...")

    try:
        # Upload with a consistent filename
        api.upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo="model.pt",  # Consistent name in repo
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload hand detection model"
        )

        # Also upload with version name
        api.upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo=model_file.name,  # Keep original name too
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {model_file.name}"
        )

        # Create model card
        model_card = f"""---
tags:
- yolov8
- image-classification
- hand-detection
- computer-vision
library_name: ultralytics
---

# Hand Detection Model (YOLOv8)

This model classifies images into three categories:
- **hand**: Close-up hand with fingers visible
- **arm**: Forearm or elbow area
- **not_hand**: Neither hand nor arm

## Usage

```python
from ultralytics import YOLO

# Load model directly from HuggingFace
model = YOLO('https://huggingface.co/{repo_id}/resolve/main/model.pt')

# Predict on an image
results = model.predict('image.jpg')

# Get predictions
if results and results[0].probs:
    probs = results[0].probs
    top_class = probs.top1  # 0=hand, 1=arm, 2=not_hand
    confidence = probs.top1conf.item()

    classes = ['hand', 'arm', 'not_hand']
    print(f"Detected: {{classes[top_class]}} ({{confidence:.1%}})")
```

## Usage in Next.js/Node.js

### Option 1: Python API Backend

```javascript
// app/api/detect/route.js (Next.js 13+ App Router)
export async function POST(request) {{
    const formData = await request.formData();
    const image = formData.get('image');

    // Call Python backend
    const response = await fetch('http://localhost:8000/predict', {{
        method: 'POST',
        body: formData
    }});

    const result = await response.json();
    return Response.json(result);
}}

// Frontend component
async function detectHand(file) {{
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/api/detect', {{
        method: 'POST',
        body: formData
    }});

    const result = await response.json();
    // result = {{ class: 'hand', confidence: 0.98 }}
    return result;
}}
```

### Option 2: Python Microservice (FastAPI)

```python
# backend/api.py
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = YOLO('https://huggingface.co/{repo_id}/resolve/main/model.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model.predict(image)
    probs = results[0].probs

    classes = ['hand', 'arm', 'not_hand']
    return {{
        "class": classes[probs.top1],
        "confidence": float(probs.top1conf),
        "all_probs": {{
            "hand": float(probs.data[0]),
            "arm": float(probs.data[1]),
            "not_hand": float(probs.data[2])
        }}
    }}
```

### Option 3: Using ONNX.js (Browser-based)

```javascript
// First convert model to ONNX (run once)
// python3 -c "from ultralytics import YOLO; YOLO('model.pt').export(format='onnx')"

import * as ort from 'onnxruntime-web';

async function detectHandBrowser(imageElement) {{
    // Load ONNX model
    const session = await ort.InferenceSession.create('/model.onnx');

    // Preprocess image to 224x224
    const tensor = preprocessImage(imageElement);

    // Run inference
    const results = await session.run({{ input: tensor }});
    const probs = results.output.data;

    // Get prediction
    const classes = ['hand', 'arm', 'not_hand'];
    const maxIdx = probs.indexOf(Math.max(...probs));

    return {{
        class: classes[maxIdx],
        confidence: probs[maxIdx],
        all_probs: {{
            hand: probs[0],
            arm: probs[1],
            not_hand: probs[2]
        }}
    }};
}}
```

## Usage in React Native

```javascript
import {{ launchImageLibrary }} from 'react-native-image-picker';

const detectHand = async () => {{
    const result = await launchImageLibrary({{ mediaType: 'photo' }});

    if (result.assets) {{
        const formData = new FormData();
        formData.append('image', {{
            uri: result.assets[0].uri,
            type: 'image/jpeg',
            name: 'photo.jpg'
        }});

        const response = await fetch('YOUR_API_URL/predict', {{
            method: 'POST',
            body: formData
        }});

        const detection = await response.json();
        console.log('Detected:', detection.class, detection.confidence);
    }}
}};
```

## Usage with cURL

```bash
# Test the model with cURL
curl -X POST -F "image=@test.jpg" http://your-api-url/predict

# Response: {{"class": "hand", "confidence": 0.98}}
```

## Usage in Swift (iOS)

```swift
import CoreML
import Vision

func detectHand(image: UIImage) {{
    // First convert YOLO to CoreML format
    // Then use in iOS app:

    guard let model = try? VNCoreMLModel(for: HandDetector().model) else {{ return }}

    let request = VNCoreMLRequest(model: model) {{ request, error in
        guard let results = request.results as? [VNClassificationObservation] else {{ return }}

        if let topResult = results.first {{
            let className = topResult.identifier // "hand", "arm", or "not_hand"
            let confidence = topResult.confidence
            print("Detected: \\(className) with \\(confidence * 100)% confidence")
        }}
    }}

    // Process image...
}}
```

## Model Details

- **Architecture**: YOLOv8s-cls
- **Classes**: 3 (hand, arm, not_hand)
- **Input Size**: 224x224
- **Training Data**: 1740 images
- **Accuracy**: >96%

## Training Details

Trained on a custom dataset with:
- 704 hand images
- 320 arm images
- 462 not_hand images

Split 80/20 for training/validation.
"""

        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card"
        )

        # Create config for easier loading
        config = """classes:
  0: hand
  1: arm
  2: not_hand
"""

        api.upload_file(
            path_or_fileobj=config.encode(),
            path_in_repo="config.yaml",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add class configuration"
        )

        print(f"\n✅ Model uploaded successfully!")
        print(f"\n📦 Use in your app:")
        print(f"```python")
        print(f"from ultralytics import YOLO")
        print(f"model = YOLO('https://huggingface.co/{repo_id}/resolve/main/model.pt')")
        print(f"results = model.predict('image.jpg')")
        print(f"```")

        print(f"\n🌐 View your model at:")
        print(f"https://huggingface.co/{repo_id}")

        return f"https://huggingface.co/{repo_id}/resolve/main/model.pt"

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\nTry running: huggingface-cli login")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace")
    parser.add_argument("--model", default="models/three_class_detector.pt", help="Model path")
    parser.add_argument("--repo", default="hand-detection-yolo", help="Repository name")
    parser.add_argument("--private", action="store_true", help="Make repository private")

    args = parser.parse_args()

    url = upload_model_to_hf(
        model_path=args.model,
        repo_name=args.repo,
        private=args.private
    )

    if url:
        print(f"\n✨ Your model URL: {url}")