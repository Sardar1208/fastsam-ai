from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
import io
import torch
import uuid
import os
import ast
from contextlib import asynccontextmanager
from typing import Optional

from fastsam import FastSAM, FastSAMPrompt
from utils.tools import convert_box_xywh_to_xyxy

# === Monkey-patch Agg canvas as in original code ===
import os
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

def _tostring_rgb(self):
    buf = self.tostring_argb()
    width, height = self.get_width_height()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
    return arr[:, :, 1:].tobytes()

FigureCanvasAgg.tostring_rgb = _tostring_rgb
# ==================================================

class Args:
    def __init__(self):
        self.model_path = "./FastSAM-x.pt"  # Update path as needed
        self.imgsz = 1024
        self.iou = 0.9
        self.conf = 0.4
        self.output = "./output/"
        self.randomcolor = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.retina = True
        self.withContours = False
        self.better_quality = False
        self.text_prompt = None
        self.point_prompt = [[0, 0]]
        self.box_prompt = [[0, 0, 0, 0]]  # Will be converted to XYXY
        self.point_label = [0]

@asynccontextmanager
async def lifespan(app: FastAPI):
    args = Args()
    model = FastSAM(args.model_path)
    app.state.model = model
    app.state.args = args
    yield
    # Cleanup if needed
    app.state.model = None
    app.state.args = None

app = FastAPI(lifespan=lifespan)

@app.post("/segment")
async def segment_image(
    image_file: UploadFile = File(...),
    text_prompt: Optional[str] = None,
    box_prompt: Optional[str] = None,
    conf: Optional[float] = 0.4,
    iou: Optional[float] = 0.9,
):
    # Read and convert image
    try:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Get model and args
    model = app.state.model
    args = app.state.args
    args.conf = conf
    args.iou = iou
    args.text_prompt = text_prompt

    # Parse box prompt if provided
    if box_prompt:
        try:
            box_list = ast.literal_eval(box_prompt)
            args.box_prompt = convert_box_xywh_to_xyxy(box_list)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid box_prompt: {e}")

    # Process image
    try:
        everything_results = model(
            image,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou
        )

        prompt_process = FastSAMPrompt(image, everything_results, device=args.device)
        ann = None
        bboxes = None

        # Determine prompt type
        if args.box_prompt and args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
            bboxes = args.box_prompt
        elif args.text_prompt:
            ann = prompt_process.text_prompt(text=args.text_prompt)
        else:
            ann = prompt_process.everything_prompt()

        # Generate output image
        temp_filename = f"{uuid.uuid4().hex}.png"
        temp_path = os.path.join("/tmp", temp_filename)
        
        prompt_process.plot(
            annotations=ann,
            output_path=temp_path,
            bboxes=bboxes,
            points=args.point_prompt,
            point_label=args.point_label,
            withContours=args.withContours,
            better_quality=args.better_quality,
        )

        with open(temp_path, "rb") as f:
            image_bytes = f.read()
        os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=image_bytes, media_type="image/png")

# To run the API: uvicorn api:app --reload