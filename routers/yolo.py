from schemas.yolo import get_image_from_bytes , detect_sample_model, add_bboxs_on_img,get_bytes_from_image
from fastapi import FastAPI, File, UploadFile, HTTPException,APIRouter,status
from fastapi.responses import StreamingResponse,JSONResponse
from ultralytics import YOLO
from typing import Optional
from loguru import logger
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import requests
import json
import io
router = APIRouter(tags=["Image Upload and analysis"], prefix="/yolo")

model_sample_model = YOLO('yolov8n.pt')
image_id_counter = 0
detection_results = {}
@router.post("/img_object_detection_to_json")
async def img_object_detection_to_json(file: UploadFile = File(...)):
    """
    Object Detection from an image.

    Args:
        file (UploadFile): The image file as an UploadFile object.
    Returns:
        dict: JSON format containing the Objects Detections, including bounding box coordinates.
    """
    global image_id_counter
    image_id_counter += 1
    # Step 1: Read the file content
    contents = await file.read()

    # Step 2: Convert the image file to an image object
    input_image = get_image_from_bytes(contents)

    # Step 3: Predict from model
    predict = detect_sample_model(input_image)

    # Modify Step 4 to include bbox coordinates
    # Here we add 'xmin', 'ymin', 'xmax', 'ymax' to the selection for the result
    detect_res = predict[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']]
    detect_res['confidence'] = detect_res['confidence'].apply(lambda x: round(x, 2))  # Optionally round confidence for better readability

    # Convert detections to a list of dictionaries
    detections = detect_res.to_dict(orient='records')
   
    # Step 5: Logs and return
    logger.info("Detection results: {}", detections)
    detection_results[image_id_counter] = detections
    return {"image_id": image_id_counter,"detections": detections}
@router.get("/get_detection_results_byID/{image_id}")
async def get_detection_results_byID(image_id: int):
    if image_id in detection_results:
        return detection_results[image_id]
    else:
        raise HTTPException(status_code=404, detail="Image ID not found")
@router.post("/img_object_detection_to_img")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    Object Detection from an image plot bbox on image

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        Image: Image in bytes with bbox annotations.
    """
    # get image from bytes
    input_image = get_image_from_bytes(file)

    # model predict
    predict = detect_sample_model(input_image)

    # add bbox on image
    final_image = add_bboxs_on_img(image = input_image, predict = predict)

    # return image in bytes format
    return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")
@router.post("/URL_object_detection_to_json/")
async def detect_objects(image_url: str):
    global image_id_counter, detection_results  
    try:
     
        response = requests.get(image_url)
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes).convert("RGB")

        predictions = detect_sample_model(image)

       
        image_id_counter += 1
        detect_res = predictions 
        detection_results[image_id_counter] = detect_res.to_dict(orient='records')  # ذخیره‌سازی نتایج با image_id

        
        return JSONResponse(content={"image_id": image_id_counter, "detections": detection_results[image_id_counter]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/URL_get_detection_results/{image_id}")
async def URL_get_detection_results(image_id: int):
    if image_id in detection_results:
        return detection_results[image_id]
    else:
        raise HTTPException(status_code=404, detail=f"Image ID {image_id} not found.")
@router.post("/URL_img_object_detection_to_img/")
async def url_img_object_detection_to_img(image_url: str):
    try:

        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to download the image from the URL provided.")

        image_bytes = BytesIO(response.content)
        input_image = Image.open(image_bytes).convert("RGB")

   
        predict = detect_sample_model(input_image)  

  
        final_image = add_bboxs_on_img(image=input_image, predict=predict)  # فرض بر این است که این تابع وجود دارد و جعبه‌های محدوده‌ای را به تصویر اضافه می‌کند

   
        final_img_bytes_io = BytesIO()
        final_image.save(final_img_bytes_io, format="JPEG")
        final_img_bytes_io.seek(0)  
        return StreamingResponse(final_img_bytes_io, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

