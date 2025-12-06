"""
Bounding Box Utilities

Shared utilities for bounding box format conversions.
Eliminates code duplication across dataset modules.
"""

from typing import Union, List, Tuple
import numpy as np


def convert_bbox_to_yolo_format(
    bbox: Union[List, Tuple, np.ndarray],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Convert bounding box from [x, y, width, height] to normalized YOLO format.
    
    YOLO format: [x_center, y_center, width, height] all normalized to [0, 1]
    
    Args:
        bbox: Bounding box in format [x, y, width, height] (top-left corner format)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List of [x_center, y_center, width, height] normalized to image dimensions
        
    Example:
        >>> bbox = [100, 150, 50, 80]  # x, y, w, h
        >>> img_w, img_h = 640, 480
        >>> yolo_bbox = convert_bbox_to_yolo_format(bbox, img_w, img_h)
        >>> # Returns: [0.1953, 0.3958, 0.0781, 0.1667]
    """
    x, y, w, h = bbox
    
    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # Normalize width and height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center, y_center, w_norm, h_norm]


def yolo_to_xyxy(
    bbox_yolo: Union[List, Tuple, np.ndarray],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Convert YOLO format to [x1, y1, x2, y2] format.
    
    Args:
        bbox_yolo: YOLO format [x_center, y_center, width, height] normalized
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List of [x1, y1, x2, y2] in pixel coordinates
    """
    xc, yc, w, h = bbox_yolo
    
    # Convert to pixel coordinates
    x_center = xc * img_width
    y_center = yc * img_height
    width = w * img_width
    height = h * img_height
    
    # Calculate corners
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2]


def yolo_to_xywh(
    bbox_yolo: Union[List, Tuple, np.ndarray],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Convert YOLO format to [x, y, width, height] format.
    
    Args:
        bbox_yolo: YOLO format [x_center, y_center, width, height] normalized
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List of [x, y, width, height] in pixel coordinates (top-left corner)
    """
    xc, yc, w, h = bbox_yolo
    
    # Convert to pixel coordinates
    x_center = xc * img_width
    y_center = yc * img_height
    width = w * img_width
    height = h * img_height
    
    # Calculate top-left corner
    x = x_center - width / 2
    y = y_center - height / 2
    
    return [x, y, width, height]
