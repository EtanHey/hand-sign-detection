#!/bin/bash

# Deploy to HuggingFace Spaces
# This script creates and deploys a HuggingFace Space with API endpoints

echo "🚀 Deploying Hand Detection Model to HuggingFace Spaces"
echo "=" * 50

# Check if user is logged in to HuggingFace
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Not logged in to HuggingFace. Please run:"
    echo "   huggingface-cli login"
    exit 1
fi

# Get username
USERNAME=$(huggingface-cli whoami | grep "username" | cut -d':' -f2 | tr -d ' ')
SPACE_NAME="hand-detection-api"
SPACE_REPO="$USERNAME/$SPACE_NAME"

echo "📦 Creating Space: $SPACE_REPO"

# Create Space repository
huggingface-cli repo create $SPACE_NAME --type space --space_sdk gradio -y

# Clone the Space
git clone https://huggingface.co/spaces/$SPACE_REPO temp_space
cd temp_space

# Copy necessary files
cp ../app_spaces.py app.py
cp ../requirements.txt .

# Create README for the Space
cat > README.md << 'EOF'
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
EOF

# Create example images directory
mkdir -p examples
echo "Add example images to the examples/ directory"

# Commit and push
git add .
git commit -m "Deploy hand detection API with Gradio interface"
git push

echo "✅ Space deployed successfully!"
echo "🌐 View your Space at: https://huggingface.co/spaces/$SPACE_REPO"
echo ""
echo "📝 Next steps:"
echo "1. Wait for the Space to build (check the Logs tab)"
echo "2. Once running, get your API endpoint:"
echo "   https://$USERNAME-$SPACE_NAME.hf.space/api/predict"
echo ""
echo "3. Update your Next.js app:"
echo "   Edit: unified-detector-client/src/lib/api/detector.ts"
echo "   Set: huggingFaceSpaceEndpoint = 'https://$USERNAME-$SPACE_NAME.hf.space/api/predict'"

# Cleanup
cd ..
rm -rf temp_space