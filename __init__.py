import torch
import numpy as np
import requests
import base64
import json
import os
import subprocess
import tempfile
import time
from io import BytesIO
from PIL import Image

from .nano_banana_pad import (
    NanaBananaPadCalculator,
    NODE_CLASS_MAPPINGS as PAD_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PAD_NODE_DISPLAY_NAME_MAPPINGS,
)


def _get_access_token(service_account_b64: str) -> tuple[str, str]:
    """Generate OAuth2 access token from base64-encoded service account JSON.

    Returns:
        Tuple of (access_token, project_id)
    """
    sa_json = base64.b64decode(service_account_b64).decode("utf-8")
    sa_data = json.loads(sa_json)

    client_email = sa_data["client_email"]
    private_key = sa_data["private_key"]
    project_id = sa_data["project_id"]

    header = {"alg": "RS256", "typ": "JWT"}
    now = int(time.time())

    payload = {
        "iss": client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": "https://oauth2.googleapis.com/token",
        "iat": now,
        "exp": now + 3600,
    }

    header_b64 = (
        base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )

    message = f"{header_b64}.{payload_b64}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as tmp:
        tmp.write(private_key)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", tmp_path, "-binary"],
            input=message.encode(),
            capture_output=True,
            check=True,
        )
        signature = base64.urlsafe_b64encode(proc.stdout).decode().rstrip("=")
    finally:
        os.unlink(tmp_path)

    jwt_token = f"{header_b64}.{payload_b64}.{signature}"

    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt_token,
        },
    )
    resp.raise_for_status()
    access_token = resp.json()["access_token"]

    return access_token, project_id


SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "OFF"},
]


class GeminiImageGenerate:
    """ComfyUI node for Gemini image generation via Vertex AI."""

    MODELS = [
        "gemini-3-pro-image-preview",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Transform this image"},
                ),
                "service_account_base64": ("STRING", {"default": ""}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "location": ("STRING", {"default": "us-central1"}),
                "aspect_ratio": ("STRING", {"default": "1:1"}),
                "resolution": ("STRING", {"default": "1K"}),
            },
            "optional": {
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image/generate"

    def _tensor_to_base64(self, image: torch.Tensor) -> str:
        """Convert a ComfyUI image tensor (BHWC, 0-1 float) to a base64 PNG string."""
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate(
        self,
        image: torch.Tensor,
        prompt: str,
        service_account_base64: str,
        model: str,
        location: str,
        aspect_ratio: str,
        resolution: str,
        image_2: torch.Tensor = None,
    ):
        # Collect all provided images
        images = [image]
        if image_2 is not None:
            images.append(image_2)

        # Build parts: text prompt + all images as inline_data
        parts = [{"text": prompt}]
        for img in images:
            img_b64 = self._tensor_to_base64(img)
            parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})

        # Authenticate via service account
        access_token, project_id = _get_access_token(service_account_base64)

        # Build request
        url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:generateContent"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution,
                },
            },
            "safety_settings": SAFETY_SETTINGS,
        }

        # Call API
        resp = requests.post(url, headers=headers, json=payload, timeout=600)
        if not resp.ok:
            import re

            body = resp.text
            body = re.sub(
                r'"data"\s*:\s*"[A-Za-z0-9+/=]{100,}"',
                '"data": "<base64 truncated>"',
                body,
            )
            print(f"[Gemini] API error {resp.status_code}: {body}")
            resp.raise_for_status()
        data = resp.json()

        # Find the image part in the response (text parts may come first)
        try:
            parts_resp = data["candidates"][0]["content"]["parts"]
        except KeyError:
            raise RuntimeError(
                f"Unexpected response structure: {json.dumps(data, indent=2)}"
            )

        output_b64 = None
        for part in parts_resp:
            if "inlineData" in part:
                output_b64 = part["inlineData"]["data"]
                break

        if output_b64 is None:
            raise ValueError(
                f"No image found in response parts: {[list(p.keys()) for p in parts_resp]}"
            )

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
