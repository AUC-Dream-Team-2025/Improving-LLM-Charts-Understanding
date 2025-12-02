"""
Chart Understanding with Qwen2.5-VL-7B-Instruct
Graduation Project - Prompting Techniques Evaluation
ENHANCED VERSION: Added Chart2Table Chain-of-Models Strategy + Strategy Selection
"""

# ============================================================================
# QUICK CONFIGURATION - Modify these before running!
# ============================================================================
# To test with 100 examples: Set NUM_SAMPLES = 100
# To test with full dataset: Set NUM_SAMPLES = None
# See main() function for more configuration options
# ============================================================================

import torch
import requests
import base64
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from qwen_vl_utils import process_vision_info  # REQUIRED: pip install qwen-vl-utils
import json
import re
import uvicorn
import traceback
from pathlib import Path
from typing import Dict, List, Any
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import io

class Config:

    # GPU Memory Management
    GPU_MEMORY_FRACTION = 0.75  # Use 75% of remaining GPU memory
    ENABLE_TF32 = True  # Faster computation on Ampere+ GPUs

    # LLM (PyTorch - same as before)
    LLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    DATASET = "jrc/cleaned-plotqa-v2"
    OUTPUT_DIR = "plotqa_safe_solver_cot_sc_v2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # PaddlePaddle API Server (microservice)
    PADDLE_API_URL = "http://localhost:8000"  # Change if running on different machine
    PADDLE_API_TIMEOUT = 30  # seconds

    # Token limits
    MAX_INPUT_TOKENS = 4096
    MAX_NEW_TOKENS = 128

    # CoT + Self-Consistency
    USE_COT = True
    FEW_SHOT_COT = True
    SC_SAMPLES = 5
    TEMP_COT = 0.7
    TOP_P_COT = 0.95

    # Debug
    VERBOSE_DEBUG = False


# ============================================================================
# MODEL SETUP (FROM FIRST CODE - CORRECT IMPLEMENTATION)
# ============================================================================

class ChartAnalyzer:
    """Main class for chart analysis using Qwen2.5-VL-7B-Instruct"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"Loading model: {model_path}")
        
        # ✅ NEW: Limit GPU memory usage BEFORE loading model
        # Set GPU memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(Config.GPU_MEMORY_FRACTION, 0)
            # Enable TF32 for faster computation (Ampere+ GPUs)
            if Config.ENABLE_TF32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        
    def generate_response(
        self, 
        image_path: str, 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,  # Lower temperature for factual chart tasks
        top_p: float = 0.9,
        few_shot_messages: List[Dict] = None
    ) -> str:
        """
        Generate response for a given image and prompt
        Uses official Qwen implementation with qwen_vl_utils
        
        Args:
            image_path: Path to the chart image
            prompt: Text prompt for the model
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more factual)
            top_p: Top-p sampling parameter
            few_shot_messages: Optional pre-built message history for multimodal few-shot
            
        Returns:
            Generated text response
        """
        
        # 1. Construct Messages
        messages = []
        
        # Add few-shot history if present
        if few_shot_messages:
            messages.extend(few_shot_messages)
            
        # Add current query
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},  # CORRECT: use 'image' key
                {"type": "text", "text": prompt}
            ]
        })
        
        # 2. Prepare Inputs (The Official Qwen Way)
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to GPU
        inputs = inputs.to(self.model.device)
        
        # 3. Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )
        
        # 4. Decode - Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response_text



# server.py
import io
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image


app = FastAPI()

print("Loading Qwen2.5-VL ChartAnalyzer...")
chart_analyzer = ChartAnalyzer("Qwen/Qwen2.5-VL-7B-Instruct")

print("Server ready.")


@app.post("/predict")
async def predict(
    question: str = Form(...),
    image: UploadFile = File(...)
):
    """
    IMAGE IS NOW MANDATORY.
    """

    try:
        # -----------------------------
        # Ensure image was uploaded
        # -----------------------------
        if image is None:
            return JSONResponse({
                "answer": "Error: An image file is required.",
                "error": "Missing image file"
            }, status_code=400)

        # -----------------------------
        # 1. Save uploaded image to disk
        # -----------------------------
        image_bytes = await image.read()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_path = "temp_upload.jpg"
        pil.save(img_path)

        # -----------------------------
        # 2. Run Qwen model
        # -----------------------------
        llm_output = chart_analyzer.generate_response(
            image_path=img_path,
            prompt=question,
            max_new_tokens=512,
            temperature=0.1
        )

        # -----------------------------
        # 5. Return result
        # -----------------------------
        return JSONResponse({
            "answer": llm_output
        })

    except Exception as e:
        traceback_str = traceback.format_exc()
        return JSONResponse({
            "answer": f"Server error: {e}",
            "error": str(e),
            "trace": traceback_str
        }, status_code=500)

if __name__ == "__main__":
    import nest_asyncio
    # Apply the patch to allow Uvicorn to run in the notebook's loop
    nest_asyncio.apply() 

    # Run with: python paddle_api_server.py
    # Or: uvicorn paddle_api_server:app --reload --host 0.0.0.0 --port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)