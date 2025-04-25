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

# # TODO: Filter and print Annotations
# def filter_top_masks(annotations, top_n: Optional[int]):
#     """Filter to keep only the top N largest masks."""
#     if top_n is None or not annotations:
#         return annotations
    
#     # Convert single annotation to list if needed
#     if not isinstance(annotations, list):
#         annotations = [annotations]
    
#     # Calculate areas and sort
#     masks_with_areas = []
#     for mask in annotations:
#         if 'segmentation' in mask:
#             area = mask['segmentation'].sum()
#             masks_with_areas.append((area, mask))
    
#     # Sort by area descending and take top N
#     masks_with_areas.sort(reverse=True, key=lambda x: x[0])
#     return [mask for area, mask in masks_with_areas[:top_n]]

import numpy as np
import torch
from typing import Optional, Union, List, Dict

def filter_top_masks(
    annotations: Union[torch.Tensor, List[Dict]], 
    top_n: Optional[int]
) -> Union[torch.Tensor, List[Dict]]:
    """Filter to keep only the top N largest masks with type consistency.
    
    Args:
        annotations: Input masks as either torch.Tensor or list of dicts
        top_n: Number of top masks to return (None returns all)
        
    Returns:
        Same type as input (tensor->tensor, list->list)
    """
    print(f"\n=== START DEBUGGING ===")
    print(f"Initial input - top_n: {top_n}, annotations type: {type(annotations)}")
    
    try:
        # 1. Type detection and conversion
        input_is_tensor = isinstance(annotations, torch.Tensor)
        device = annotations.device if input_is_tensor else None
        
        if input_is_tensor:
            print("Input is torch.Tensor - converting to numpy")
            annotations = annotations.cpu().numpy()
            print(f"Converted to numpy array of shape: {annotations.shape}")
            annotations = [{'segmentation': mask} for mask in annotations]
            print(f"Created {len(annotations)} mask dictionaries")

        if top_n is None or not annotations:
            print("Returning original annotations (no filtering needed)")
            return annotations

        # 2. Process masks
        print(f"\nProcessing {len(annotations)} annotations:")
        masks_with_areas = []
        
        for i, mask in enumerate(annotations):
            try:
                print(f"\n--- Annotation {i} ---")
                print(f"Mask keys: {mask.keys() if isinstance(mask, dict) else 'Not a dict'}")
                
                if not isinstance(mask, dict):
                    print(f"Skipping non-dict annotation (type: {type(mask)})")
                    continue
                    
                if 'segmentation' not in mask:
                    print("Skipping mask without segmentation")
                    continue
                    
                # Validate and convert segmentation
                seg = mask['segmentation']
                if isinstance(seg, torch.Tensor):
                    seg = seg.cpu().numpy()
                elif not isinstance(seg, np.ndarray):
                    print(f"Invalid segmentation type: {type(seg)}")
                    continue
                
                # Ensure binary mask
                seg = seg.astype(np.uint8)
                unique_vals = np.unique(seg)
                if len(unique_vals) > 2 or not (0 in unique_vals and 1 in unique_vals):
                    print(f"Invalid mask values: {unique_vals}")
                    continue
                
                area = seg.sum()
                print(f"Valid mask - area: {area}")
                masks_with_areas.append((area, mask))
                
            except Exception as e:
                print(f"ERROR processing mask {i}: {str(e)}")
                continue

        print(f"\nValid masks found: {len(masks_with_areas)}")
        
        if not masks_with_areas:
            print("No valid masks found, returning empty list")
            return torch.tensor([]) if input_is_tensor else []
            
        # 3. Sort and select top N
        masks_with_areas.sort(reverse=True, key=lambda x: x[0])
        result = [mask for area, mask in masks_with_areas[:top_n]]
        
        print("\nTop masks after sorting:")
        for i, (area, _) in enumerate(masks_with_areas[:min(top_n, len(masks_with_areas))]):
            print(f"Top {i}: area={area}")
        
        # 4. Type-consistent return
        if input_is_tensor:
            print("Converting results back to torch.Tensor")
            tensor_result = torch.stack([
                torch.from_numpy(m['segmentation']).to(device) 
                for m in result
            ])
            print(f"Returning torch.Tensor of shape {tensor_result.shape}")
            return tensor_result
        
        print(f"Returning list of {len(result)} dicts")
        return result
        
    except Exception as e:
        print(f"FATAL ERROR in filter_top_masks: {str(e)}")
        raise



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
    top_masks: Optional[str] = None
):
    # Convert top_masks to integer if provided
    top_n = int(top_masks) if top_masks is not None and top_masks.isdigit() else None
    print(f"top_masks received: {top_masks} -> converted: {top_n}")  # Debug

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

        # After getting annotations but before plotting:
        # if top_n is not None:
        ann = filter_top_masks(ann, 1)
        
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


# python -m uvicorn api:app --reload
# curl -X POST -F "image_file=@screenshot.png" -F "top_masks=3" -F "min_area=100" http://localhost:8000/segment --output output.png
