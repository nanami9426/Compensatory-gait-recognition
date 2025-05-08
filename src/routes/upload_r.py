from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from datetime import datetime

upload_route = APIRouter()

SAVE_DIR = "../uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

@upload_route.post("/upload/")
async def upload_file(file: UploadFile = File(...), file_type: str = Form(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_filename = f"{timestamp}_{file.filename}"
        image_path = os.path.join(SAVE_DIR, image_filename)
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        # 保存文本
        text_filename = f"{timestamp}_{file.filename}.txt"
        text_path = os.path.join(SAVE_DIR, text_filename)
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(file_type)

        return JSONResponse(content={
            "message": "上传成功",
            "image_path": image_path,
            "text_path": text_path
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
