import cv2 
import numpy as np
from re import DOTALL, finditer
import base64
from io import BytesIO
from PIL import Image
from imantics import Mask


def fit_polygon_from_mask(mask):
    # return polygon vertices
    polygons = Mask(mask).polygons()
    if len(polygons.points) > 1: # find the largest polygon
        areas = []
        for points in polygons.points:
            points = points.reshape((-1, 1, 2)).astype(np.int32)
            img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            img = cv2.fillPoly(img, [points], color=[0, 255, 0])
            mask = img[:, :, 1] > 0
            area = np.count_nonzero(mask)
            areas.append(area)
        areas = np.array(areas)
        largest_idx = np.argmax(areas)
        points = polygons.points[largest_idx]
    else:
        points = polygons.points[0]
    return points             
    

# same function in simulation/sim_utils.py
def fit_circle_from_mask(mask_image):
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No contours found in the mask image.")
        return None
    max_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(max_contour)

    center = (x, y)
    return center, radius


def is_mask_truncated(mask):
    if np.any(mask[0, :] == 1) or np.any(mask[-1, :] == 1):  # Top or bottom rows
        return True
    if np.any(mask[:, 0] == 1) or np.any(mask[:, -1] == 1):  # Left or right columns
        return True
    return False


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0.0
    
    return iou


def find_json_response(full_response):
    extracted_responses = list(
        finditer(r"({[^}]*$|{.*})", full_response, flags=DOTALL)
    )
   
    if not extracted_responses:
        print(
            f"Unable to find any responses of the matching type dictionary: `{full_response}`"
        )
        return None

    if len(extracted_responses) > 1:
        print("Unexpected response > 1, continuing anyway...", extracted_responses)

    extracted_response = extracted_responses[0]
    extracted_str = extracted_response.group(0)
    return extracted_str


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")