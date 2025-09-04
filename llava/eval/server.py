# server.py
import os
import io
import zipfile
import torch
import uvicorn
from typing import List, Optional, Literal
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# ---- your LLaVA imports ----
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


API_KEY = os.environ.get("NAVILA_API_KEY", "dev-key-change-me")
DEFAULT_MODEL_PATH = os.environ.get("NAVILA_MODEL_PATH", "/PATH/checkpoints/vila-long-8b-8f-scanqa-rxr-simplified-v8-lowcam")
_navila_model_base_env = os.environ.get("NAVILA_MODEL_BASE")
DEFAULT_MODEL_BASE = _navila_model_base_env if _navila_model_base_env else None

app = FastAPI(title="NaVILA Inference Server", version="1.0")

class ActionOut(BaseModel):
    # Structured action extracted from text (very simple heuristic here; customize to your schema)
    action: Literal["turn_left", "turn_right", "move_forward", "stop", "unknown"]
    value: Optional[float] = None  # degrees for turn, meters for forward
    raw_text: str

class ModelBundle:
    def __init__(self, model_path: str, model_base: Optional[str], gpu_id: int = 0):
        disable_torch_init()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_name, model_base
        )
        self.stop_str = conv_templates["llama_3"].sep if conv_templates["llama_3"].sep_style != SeparatorStyle.TWO else conv_templates["llama_3"].sep2

    def run(self, images: List[Image.Image], query: str, num_video_frames: int, temperature: float) -> str:
        conv = conv_templates["llama_3"].copy()
        image_token = "<image>\n"
        qs = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {image_token * (num_video_frames-1)}, and current observation <image>\n. Your assigned task is: "{query}" '
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            f"degree, moving forward a certain distance, or stop if the task is completed."
        )
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        device = self.model.device
        images_tensor = process_images(images, self.image_processor, self.model.config).to(device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        stopping_criteria = [KeywordsStoppingCriteria([self.stop_str], self.tokenizer, input_ids)]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor.half(),
                do_sample=(temperature > 0.0),
                temperature=temperature,
                max_new_tokens=256,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(self.stop_str):
            outputs = outputs[: -len(self.stop_str)].strip()
        return outputs

# Lazy global (so we can pick GPU per request if you want)
_model_cache = {}

def get_model(gpu_id: int) -> ModelBundle:
    key = (DEFAULT_MODEL_PATH, DEFAULT_MODEL_BASE, gpu_id)
    if key not in _model_cache:
        _model_cache[key] = ModelBundle(DEFAULT_MODEL_PATH, DEFAULT_MODEL_BASE, gpu_id=gpu_id)
    return _model_cache[key]

def ensure_auth(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: bad API key")

def _load_images_from_uploads(files: List[UploadFile]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for f in files:
        content = f.file.read()
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            images.append(img)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Could not read image: {f.filename}")
    return images

def _load_images_from_zip(zf: UploadFile) -> List[Image.Image]:
    try:
        data = zf.file.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = sorted([n for n in z.namelist() if n.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp"))])
            if not names:
                raise ValueError("No images in zip")
            images = []
            for name in names:
                with z.open(name) as fp:
                    images.append(Image.open(io.BytesIO(fp.read())).convert("RGB"))
            return images
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad zip: {e}")

def extract_action(text: str) -> ActionOut:
    t = text.lower()
    # Very naive parsing—replace with your production regex/LLM tool
    if "stop" in t:
        return ActionOut(action="stop", value=None, raw_text=text)
    import re
    m = re.search(r"(turn|rotate)\s+(left|right)\s+(\d+(\.\d+)?)", t)
    if m:
        deg = float(m.group(3))
        return ActionOut(action=f"turn_{m.group(2)}", value=deg, raw_text=text)
    m = re.search(r"(move|go)\s+(forward)\s+(\d+(\.\d+)?)", t)
    if m:
        meters = float(m.group(3))
        return ActionOut(action="move_forward", value=meters, raw_text=text)
    return ActionOut(action="unknown", value=None, raw_text=text)

@app.post("/infer", response_model=ActionOut)
async def infer(
    query: str = Form(...),
    num_video_frames: int = Form(8),
    temperature: float = Form(0.0),
    gpu_id: int = Form(0),
    # Send either multiple images OR a zip of frames
    images: Optional[List[UploadFile]] = None,
    frames_zip: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None),
):
    ensure_auth(x_api_key)

    if not images and not frames_zip:
        raise HTTPException(status_code=400, detail="Provide images[] or frames_zip")

    try:
        if frames_zip:
            imgs = _load_images_from_zip(frames_zip)
        else:
            imgs = _load_images_from_uploads(images)
        if len(imgs) == 0:
            raise HTTPException(status_code=400, detail="No valid images provided")
        # Truncate or pad to num_video_frames (simple policy)
        if len(imgs) > num_video_frames:
            imgs = imgs[-num_video_frames:]
        elif len(imgs) < num_video_frames:
            imgs = [imgs[0]] * (num_video_frames - len(imgs)) + imgs

        model = get_model(gpu_id)
        text = model.run(imgs, query, num_video_frames, temperature)
        return extract_action(text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "cuda": torch.cuda.is_available(), "gpus": torch.cuda.device_count()})

if __name__ == "__main__":
    # Bind to 0.0.0.0 only if you've locked down firewall/IPs. Otherwise bind to the UW static IP.
    host = os.environ.get("BIND_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8009"))
    uvicorn.run("server:app", host=host, port=port, workers=1)
