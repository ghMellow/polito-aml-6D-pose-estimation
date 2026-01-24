"""
Bounding box utilities for format conversions and cropping helpers.
"""

from typing import Union, List, Tuple
import numpy as np
import cv2
from PIL import Image

from config import Config


def _normalize_bbox_and_dims(
    image: Union[np.ndarray, Image.Image],
    bbox: Union[np.ndarray, list, Tuple[float, float, float, float]]
):
    """Return (x, y, w, h, is_numpy, width_img, height_img) after validation."""
    is_numpy = isinstance(image, np.ndarray)

    if isinstance(bbox, np.ndarray):
        x, y, w, h = bbox.tolist()
    else:
        x, y, w, h = bbox

    if w <= 0 or h <= 0:
        raise ValueError(
            f"Invalid bbox dimensions: width={w}, height={h}. Bbox must have positive width and height."
        )

    if is_numpy:
        height_img, width_img = image.shape[:2]
    else:
        width_img, height_img = image.size

    return x, y, w, h, is_numpy, width_img, height_img

def crop_and_pad(img, bbox, output_size, margin=0.0):
    """
    Center crop around bbox with optional padding. Output is always square.
    """
    x, y, w, h = bbox
    x_c, y_c = x + w / 2, y + h / 2
    w_m = w * (1 + margin)
    h_m = h * (1 + margin)
    x = x_c - w_m / 2
    y = y_c - h_m / 2
    w = w_m
    h = h_m
    H, W = img.shape[:2]
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = int(np.ceil(x + w))
    y2 = int(np.ceil(y + h))
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    x1 += pad_left
    y1 += pad_top
    x2 += pad_left
    y2 += pad_top
    crop = img_padded[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    return crop_resized


def convert_bbox_to_yolo_format(
    bbox: Union[List, Tuple, np.ndarray],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Convert bbox from [x, y, width, height] (top-left) to normalized YOLO
    format [x_center, y_center, width, height] in [0, 1].
    """
    x, y, w, h = bbox
    
    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # Normalize width and height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center, y_center, w_norm, h_norm]


def convert_bbox_xywh_to_xyxy(bbox_xywh: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [x, y, width, height] to [x1, y1, x2, y2] format.
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xywh[:, 0] + bbox_xywh[:, 2]
    bbox_xyxy[:, 3] = bbox_xywh[:, 1] + bbox_xywh[:, 3]
    return bbox_xyxy

def crop_bbox_optimized(img, bbox_xyxy, margin=0.15, output_size=(224, 224)):
    """
    Crop bbox from xyxy format (YOLO output) with margin and resize.
    """
    x1, y1, x2, y2 = bbox_xyxy
    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
    return crop_and_pad(img, bbox_xywh, output_size, margin=margin)


def crop_image_from_bbox(
    image: Union[np.ndarray, Image.Image],
    bbox: Union[np.ndarray, list],
    margin: float = None,
    output_size: Tuple[int, int] = None
) -> Image.Image:
    """
    Crop image using a bounding box with optional margin and resize to output size.
    """
    if margin is None:
        margin = Config.POSE_CROP_MARGIN
    if output_size is None:
        img_size = Config.POSE_IMAGE_SIZE
        output_size = (img_size, img_size)
    x, y, w, h, is_numpy, width_img, height_img = _normalize_bbox_and_dims(image, bbox)

    margin_w = int(w * margin)
    margin_h = int(h * margin)

    x1 = max(0, int(x - margin_w))
    y1 = max(0, int(y - margin_h))
    x2 = min(width_img, int(x + w + margin_w))
    y2 = min(height_img, int(y + h + margin_h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}. "
            f"Original bbox: x={x}, y={y}, w={w}, h={h}, margin={margin}"
        )

    if is_numpy:
        cropped_np = image[y1:y2, x1:x2]
        if cropped_np.dtype != np.uint8:
            cropped_np = cropped_np.astype(np.uint8)
        cropped = Image.fromarray(cropped_np)
    else:
        cropped = image.crop((x1, y1, x2, y2))

    return cropped.resize(output_size, Image.LANCZOS)