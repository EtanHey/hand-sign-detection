# 🤚 Hand Detection API Documentation

## API Endpoint
**Base URL**: `https://etanhey-hand-detection-api.hf.space`

## Overview
This API provides real-time hand/arm detection using a YOLOv8 model with 96.3% accuracy. It classifies images into three categories:
- `hand` - Human hands
- `arm` - Human arms (without visible hands)
- `not_hand` - Neither hands nor arms

## Authentication
No authentication required - the API is publicly accessible.

## Rate Limits
- HuggingFace Spaces have automatic rate limiting
- Recommended: Max 10 requests per second per IP

---

## 🔌 API Methods

### Method 1: Gradio Client API (Recommended)

#### Python
```python
from gradio_client import Client
import json

# Initialize client
client = Client("EtanHey/hand-detection-api")

# Make prediction
result = client.predict(
    "path/to/image.jpg",  # Local file path
    api_name="/predict"
)

print(result)
# Output: ('hand', 0.963, {'hand': 0.963, 'arm': 0.025, 'not_hand': 0.012})
```

#### JavaScript/Node.js
```javascript
import { client } from "@gradio/client";

const app = await client("EtanHey/hand-detection-api");
const result = await app.predict("/predict", ["path/to/image.jpg"]);

console.log(result.data);
// Output: ['hand', 0.963, {hand: 0.963, arm: 0.025, not_hand: 0.012}]
```

### Method 2: Direct HTTP API

#### Endpoint
```
POST https://etanhey-hand-detection-api.hf.space/run/predict
```

#### Request Format
```json
{
  "data": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  ]
}
```

#### Response Format
```json
{
  "data": [
    {
      "label": "## 🎯 Detection: hand\n\n**Confidence:** 96.3%",
      "type": "markdown"
    },
    {
      "label": "hand",
      "confidences": [
        {"label": "hand", "confidence": 0.963},
        {"label": "arm", "confidence": 0.025},
        {"label": "not_hand", "confidence": 0.012}
      ]
    },
    {
      "class": "hand",
      "confidence": 0.963,
      "probabilities": {
        "hand": 0.963,
        "arm": 0.025,
        "not_hand": 0.012
      }
    }
  ],
  "is_generating": false,
  "duration": 0.234,
  "average_duration": 0.189
}
```

---

## 📚 Usage Examples

### Python with Requests
```python
import requests
import base64
import json

def detect_hand(image_path):
    # Read and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    # Prepare request
    url = "https://etanhey-hand-detection-api.hf.space/run/predict"
    payload = {
        "data": [f"data:image/jpeg;base64,{image_base64}"]
    }

    # Make request
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        # Extract the detection result from the third element
        detection = result["data"][2]
        return detection
    else:
        return {"error": f"Request failed: {response.status_code}"}

# Usage
result = detect_hand("hand_photo.jpg")
print(f"Detected: {result['class']} with {result['confidence']:.1%} confidence")
```

### JavaScript/TypeScript (Browser/Node.js)
```typescript
interface DetectionResult {
  class: 'hand' | 'arm' | 'not_hand';
  confidence: number;
  probabilities: {
    hand: number;
    arm: number;
    not_hand: number;
  };
}

async function detectHand(imageFile: File): Promise<DetectionResult> {
  // Convert image to base64
  const base64 = await new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.readAsDataURL(imageFile);
  });

  // Make API request
  const response = await fetch('https://etanhey-hand-detection-api.hf.space/run/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      data: [base64]
    })
  });

  const result = await response.json();

  // Extract detection from response
  return result.data[2] as DetectionResult;
}

// Usage
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const file = fileInput.files[0];
const detection = await detectHand(file);
console.log(`Detected: ${detection.class} (${(detection.confidence * 100).toFixed(1)}%)`);
```

### cURL
```bash
# First, convert image to base64
base64_image=$(base64 -i image.jpg)

# Make API request
curl -X POST https://etanhey-hand-detection-api.hf.space/run/predict \
  -H "Content-Type: application/json" \
  -d "{\"data\": [\"data:image/jpeg;base64,${base64_image}\"]}" \
  | jq '.data[2]'
```

### React Native
```javascript
import { launchImageLibrary } from 'react-native-image-picker';

const detectHandInImage = async () => {
  // Pick image from gallery
  const result = await launchImageLibrary({
    mediaType: 'photo',
    includeBase64: true
  });

  if (result.assets && result.assets[0].base64) {
    const response = await fetch('https://etanhey-hand-detection-api.hf.space/run/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data: [`data:image/jpeg;base64,${result.assets[0].base64}`]
      })
    });

    const data = await response.json();
    const detection = data.data[2];

    Alert.alert(
      'Detection Result',
      `${detection.class} (${(detection.confidence * 100).toFixed(1)}% confidence)`
    );
  }
};
```

---

## 🎨 Response Data Structure

The API returns three data elements in the `data` array:

1. **Markdown Result** (index 0): Human-readable formatted result
2. **Label Confidences** (index 1): Gradio Label component format
3. **JSON Result** (index 2): Developer-friendly JSON format

For programmatic use, always use `data[2]` which contains:
```typescript
{
  class: string,        // The detected class
  confidence: number,   // Confidence score (0-1)
  probabilities: {      // All class probabilities
    hand: number,
    arm: number,
    not_hand: number
  }
}
```

---

## 🖼️ Image Requirements

- **Formats**: JPEG, PNG, WebP, BMP
- **Size**: No strict limit, but images are resized to 224x224 for processing
- **Quality**: Higher quality images yield better results
- **Content**: Best results with clear, well-lit images

---

## ⚠️ Error Handling

### Common Status Codes
- `200`: Success
- `422`: Invalid input format
- `429`: Rate limit exceeded
- `500`: Server error
- `503`: Model loading or temporary unavailability

### Error Response Format
```json
{
  "error": "Error message",
  "status": 422
}
```

### Handling Errors in Code
```python
try:
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## 🚀 Performance Tips

1. **Batch Processing**: Process multiple images sequentially with small delays
2. **Image Optimization**: Resize large images before sending (max 1024x1024 recommended)
3. **Caching**: Cache results for identical images
4. **Retry Logic**: Implement exponential backoff for failed requests
5. **Connection Pooling**: Reuse HTTP connections for multiple requests

### Example with Retry Logic
```python
import time
from typing import Optional

def detect_with_retry(image_path: str, max_retries: int = 3) -> Optional[dict]:
    for attempt in range(max_retries):
        try:
            result = detect_hand(image_path)
            if "error" not in result:
                return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Retry {attempt + 1} after {wait_time}s...")
            time.sleep(wait_time)
    return None
```

---

## 🔧 Integration Examples

### Next.js App Router
```typescript
// app/api/detect/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const { image } = await request.json();

  const response = await fetch('https://etanhey-hand-detection-api.hf.space/run/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: [image] })
  });

  const result = await response.json();
  return NextResponse.json(result.data[2]);
}
```

### Flask Backend
```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    image_data = request.json.get('image')

    api_response = requests.post(
        'https://etanhey-hand-detection-api.hf.space/run/predict',
        json={'data': [image_data]}
    )

    if api_response.status_code == 200:
        result = api_response.json()
        return jsonify(result['data'][2])
    else:
        return jsonify({'error': 'Detection failed'}), 500
```

---

## 📊 Model Information

- **Architecture**: YOLOv8s-cls (Classification)
- **Training Data**: 1,740 hand-labeled images
- **Accuracy**: 96.3% on validation set
- **Classes**: 3 (hand, arm, not_hand)
- **Model Size**: ~10MB
- **Inference Time**: ~50-200ms per image
- **GPU**: Not required (CPU inference supported)

---

## 🔗 Related Resources

- **HuggingFace Space**: https://huggingface.co/spaces/EtanHey/hand-detection-api
- **Model Card**: https://huggingface.co/EtanHey/hand-sign-detection
- **GitHub Repository**: https://github.com/EtanHey/hand-sign-detection
- **Next.js Client**: `/Users/etanheyman/Desktop/Gits/unified-detector-client`

---

## 📝 Changelog

### v1.0.0 (2024-09-15)
- Initial API deployment
- YOLOv8 model with 96.3% accuracy
- Support for hand/arm/not_hand classification
- Gradio interface with API endpoints

---

## 🤝 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/EtanHey/hand-sign-detection/issues)
- Check the [HuggingFace Space logs](https://huggingface.co/spaces/EtanHey/hand-detection-api/logs)
- Contact: [Your contact info]

---

## 📄 License

This API is provided as-is for educational and research purposes. Please refer to the model card for licensing information.