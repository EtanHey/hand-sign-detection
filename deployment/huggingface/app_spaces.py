"""
HuggingFace Spaces App for Hand/Arm Detection
Provides both Gradio UI and API endpoints
Model: https://huggingface.co/EtanHey/hand-sign-detection
"""

import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json
from typing import Dict, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
MODEL_PATH = "https://huggingface.co/EtanHey/hand-sign-detection/resolve/main/model.pt"
model = None

def load_model():
    """Load YOLO model from HuggingFace"""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# Load model on startup
load_model()

# Class names (alphabetical order as YOLO expects)
CLASS_NAMES = ['arm', 'hand', 'not_hand']
CLASS_LABELS = {
    'arm': '💪 Arm',
    'hand': '✋ Hand',
    'not_hand': '❌ Not Hand/Arm'
}

def process_image(image: Image.Image) -> Dict[str, Any]:
    """Process image and return detection results"""
    if model is None:
        return {
            "error": "Model not loaded",
            "class": "unknown",
            "confidence": 0.0,
            "probabilities": {"hand": 0, "arm": 0, "not_hand": 0}
        }

    try:
        # Convert PIL image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Run inference
        results = model.predict(image, verbose=False)

        if not results or not results[0].probs:
            return {
                "class": "not_hand",
                "confidence": 0.0,
                "probabilities": {"hand": 0, "arm": 0, "not_hand": 1.0}
            }

        # Extract probabilities
        probs = results[0].probs
        top_class_idx = probs.top1
        top_confidence = float(probs.top1conf)

        # Build probability dictionary
        probabilities = {
            "hand": float(probs.data[1]),  # Index 1
            "arm": float(probs.data[0]),   # Index 0
            "not_hand": float(probs.data[2]) # Index 2
        }

        return {
            "class": CLASS_NAMES[top_class_idx],
            "confidence": top_confidence,
            "probabilities": probabilities,
            "label": CLASS_LABELS[CLASS_NAMES[top_class_idx]]
        }

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {
            "error": str(e),
            "class": "error",
            "confidence": 0.0,
            "probabilities": {"hand": 0, "arm": 0, "not_hand": 0}
        }

def gradio_predict(image: Image.Image) -> Tuple[str, Dict, str]:
    """Gradio interface prediction function"""
    if image is None:
        return "Please upload an image", {}, ""

    # Process the image
    result = process_image(image)

    # Format output
    if "error" in result:
        return f"Error: {result['error']}", {}, ""

    # Create confidence bars
    confidence_scores = {
        "✋ Hand": result["probabilities"]["hand"],
        "💪 Arm": result["probabilities"]["arm"],
        "❌ Neither": result["probabilities"]["not_hand"]
    }

    # Create detailed output
    main_label = result["label"]
    confidence = result["confidence"]

    output_text = f"""
    ## Detection Result

    **Detected:** {main_label}
    **Confidence:** {confidence:.1%}

    ### Detailed Probabilities:
    - Hand: {result['probabilities']['hand']:.1%}
    - Arm: {result['probabilities']['arm']:.1%}
    - Not Hand/Arm: {result['probabilities']['not_hand']:.1%}

    ### Understanding the Classes:
    - **Hand**: Close-up view with fingers visible
    - **Arm**: Forearm or elbow area without fingers
    - **Not Hand/Arm**: Neither hand nor arm detected
    """

    # Create JSON output for developers
    json_output = json.dumps(result, indent=2)

    return output_text, confidence_scores, json_output

# API prediction function for Gradio's built-in API
def api_predict(image: Image.Image) -> Dict[str, Any]:
    """API function that returns raw results for API access"""
    if image is None:
        return {"error": "No image provided"}

    return process_image(image)

# Gradio Interface
def create_gradio_interface():
    """Create the Gradio interface"""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .output-class {
        font-size: 24px;
        font-weight: bold;
    }
    """

    # Create interface (no examples to avoid file errors)
    interface = gr.Interface(
        fn=gradio_predict,
        inputs=[
            gr.Image(
                type="pil",
                label="Upload Image",
                sources=["upload", "webcam", "clipboard"]
            )
        ],
        outputs=[
            gr.Markdown(label="Detection Result"),
            gr.Label(label="Confidence Scores", num_top_classes=3),
            gr.JSON(label="API Response (for developers)")
        ],
        title="🤚 Hand/Arm Detection AI",
        description="""
        Upload an image or use your webcam to detect hands and arms.

        **Model:** YOLOv8 trained on 1,740 images | **Accuracy:** 96.3%

        **API Access:** Use Gradio's built-in API endpoints for programmatic access.
        """,
        article="""
        ### About
        This model distinguishes between:
        - **Hands**: Close-up views with visible fingers
        - **Arms**: Forearm/elbow areas without fingers
        - **Neither**: Images without hands or arms

        ### API Usage
        ```python
        from gradio_client import Client

        # Connect to the API
        client = Client("https://huggingface.co/spaces/EtanHey/hand-detection-api")

        # Make prediction
        result = client.predict(
            image="path/to/your/image.jpg",
            api_name="/predict"
        )
        print(result)
        ```

        ### Model Card
        View the full model details at [HuggingFace Model Hub](https://huggingface.co/EtanHey/hand-sign-detection)
        """,
        css=custom_css,
        theme=gr.themes.Soft()
    )

    return interface

# Create and launch Gradio interface
if __name__ == "__main__":
    # Create the main interface
    interface = create_gradio_interface()

    # Create API interface for programmatic access
    api_interface = gr.Interface(
        fn=api_predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.JSON(),
        title="Hand Detection API"
    )

    # Combine both interfaces in a tabbed interface
    demo = gr.TabbedInterface(
        [interface, api_interface],
        ["Web Interface", "API"],
        title="🤚 Hand/Arm Detection AI"
    )

    # Launch on default HuggingFace Spaces port (7860)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )