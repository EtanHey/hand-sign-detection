#!/usr/bin/env python3
"""
FastAPI backend for hand detection
Loads model directly from HuggingFace
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from typing import Dict, Any
import uvicorn

app = FastAPI(title="Hand Detection API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model from HuggingFace on startup
print("Loading model from HuggingFace...")
model = YOLO('https://huggingface.co/EtanHey/hand-detection-3class/resolve/main/model.pt')
print("Model loaded successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "hand-detection-3class",
        "source": "https://huggingface.co/EtanHey/hand-detection-3class",
        "classes": ["hand", "arm", "not_hand"]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict hand/arm/not_hand from uploaded image

    Returns:
        - class: detected class name
        - confidence: confidence score
        - all_probs: probabilities for all classes
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Run inference
        results = model.predict(image, verbose=False)

        if not results or not results[0].probs:
            return {
                "error": "No detection results",
                "class": "not_hand",
                "confidence": 0.0,
                "all_probs": {
                    "hand": 0.0,
                    "arm": 0.0,
                    "not_hand": 1.0
                }
            }

        # Extract probabilities
        probs = results[0].probs
        # YOLO uses alphabetical order for classes!
        classes = ['arm', 'hand', 'not_hand']  # Index 0=arm, 1=hand, 2=not_hand

        # Get predictions
        top_class_idx = probs.top1
        top_confidence = float(probs.top1conf)

        # Build response
        return {
            "class": classes[top_class_idx],
            "confidence": top_confidence,
            "all_probs": {
                "hand": float(probs.data[1]),  # Index 1 = hand
                "arm": float(probs.data[0]),   # Index 0 = arm
                "not_hand": float(probs.data[2])  # Index 2 = not_hand
            },
            "is_hand": top_class_idx == 1,  # hand is index 1
            "is_arm": top_class_idx == 0,   # arm is index 0
            "threshold_passed": top_confidence > 0.7
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {
            "error": str(e),
            "class": "error",
            "confidence": 0.0,
            "all_probs": {
                "hand": 0.0,
                "arm": 0.0,
                "not_hand": 0.0
            }
        }

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> list[Dict[str, Any]]:
    """Process multiple images at once"""
    results = []
    for file in files:
        result = await predict(file)
        results.append(result)
    return results

if __name__ == "__main__":
    # Run with: python3 api.py
    # Or with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )