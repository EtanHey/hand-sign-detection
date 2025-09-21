"""
Production API server for hand-sign-detection
Deploys to Render.com
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image
import uvicorn
import os
from pathlib import Path

app = FastAPI()

# Enable CORS for all origins (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the latest model
def get_latest_model():
    """Find the latest unified model"""
    models_dir = Path("models")
    if models_dir.exists():
        models = sorted(models_dir.glob("unified_v*.pt"), reverse=True)
        if models:
            return str(models[0])

    # Fallback to default
    if os.path.exists("models/unified_detector.pt"):
        return "models/unified_detector.pt"

    # Use HuggingFace if no local model
    return 'https://huggingface.co/EtanHey/hand-sign-detection/resolve/main/model.pt'

# Initialize model
model_path = get_latest_model()
print(f"Loading model: {model_path}")
model = YOLO(model_path)
model_name = Path(model_path).stem if os.path.exists(model_path) else "huggingface_model"

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "hand-sign-detection",
        "model": model_name
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """Detect hand/arm/not_hand in uploaded image"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run detection
        results = model(image)
        probs = results[0].probs

        # Get class names (alphabetical: arm, hand, not_hand)
        class_names = ['arm', 'hand', 'not_hand']

        # Get top prediction
        top_idx = probs.top1
        top_class = class_names[top_idx]
        confidence = float(probs.top1conf)

        # Get all probabilities
        probabilities = {
            'hand': float(probs.data[1]),
            'arm': float(probs.data[0]),
            'not_hand': float(probs.data[2])
        }

        return {
            "class": top_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "model": model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)