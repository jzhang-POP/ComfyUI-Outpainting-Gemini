import torch
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image

from .nano_banana_pad import (
    NanaBananaPadCalculator,
    NODE_CLASS_MAPPINGS as PAD_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PAD_NODE_DISPLAY_NAME_MAPPINGS,
)


class GeminiImageGenerate:
    """ComfyUI node for Gemini 3 image generation."""
    
    MODELS = [
        "gemini-3-pro-image-preview",
        "gemini-3-nano-image-preview",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Transform this image"}),
                "api_key": ("STRING", {"default": ""}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "aspect_ratio": ("STRING", {"default": "1:1"}),
                "resolution": ("STRING", {"default": "1K"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image/generate"

    def generate(self, image: torch.Tensor, prompt: str, api_key: str, model: str, aspect_ratio: str, resolution: str):
        # Convert ComfyUI tensor (BHWC, 0-1 float) to base64 PNG
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Build request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                ]
            }],
            "generationConfig": {
                "responseModalities": ["Image"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution,
                },
            },
        }
        
        # Call API
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract output image
        output_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        output_bytes = base64.b64decode(output_b64)
        output_pil = Image.open(BytesIO(output_bytes)).convert("RGB")
        
        # Convert back to ComfyUI tensor (BHWC, 0-1 float)
        output_np = np.array(output_pil).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_np).unsqueeze(0)
        
        return (output_tensor,)


NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerate": GeminiImageGenerate,
}
NODE_CLASS_MAPPINGS.update(PAD_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerate": "Gemini Image Generate",
}
NODE_DISPLAY_NAME_MAPPINGS.update(PAD_NODE_DISPLAY_NAME_MAPPINGS)