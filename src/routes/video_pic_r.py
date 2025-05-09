from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from datetime import datetime
import aiohttp
from openai import OpenAI
import base64
from dotenv import load_dotenv
load_dotenv()
video_pic_route = APIRouter()

SAVE_DIR = "../uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key=os.getenv("API_KEY_REQ"),
)


async def process_file(file, file_type):
    assert file_type in ['pic', 'video']
    if file_type == 'pic':
        content = await file.read()
        base64_image = base64.b64encode(content).decode("utf8")
        image_url = f"data:{file.content_type};base64,{base64_image}"

        prompt = '''
            通过这张图片从身体姿势可以看到哪些典型的代偿特征？
            '''
        response = client.responses.create(
            model = "gpt-4.1",
            input = [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                    }
                ]
            }]
        )
        return response.output_text
    elif file_type == "video":
        return


@video_pic_route.post("/upload/")
async def upload_file(file: UploadFile = File(...), file_type: str = Form(...)):
    try:
        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # image_filename = f"{timestamp}_{file.filename}"
        # image_path = os.path.join(SAVE_DIR, image_filename)
        # with open(image_path, "wb") as buffer:
        #     buffer.write(await file.read())


        res = await process_file(file, file_type)
        return res

        return JSONResponse(content={
            "message": "上传成功",
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
