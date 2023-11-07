import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import base64
import os, glob

from pydantic import BaseModel 


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ImageData(BaseModel):
    data: list

data_path = 'data.jsonl'
image_dir = '/Users/phil/Desktop/screenshots/multimodal'

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_initial_data():
    if os.path.exists(data_path):
        image_data = load_jsonl(data_path)
    else: 
        image_data = []

    image_paths = glob.glob(os.path.join(image_dir, '*'))
    existing_image_paths = set([x['image_path'] for x in image_data])
    for image_path in image_paths:
        if os.path.basename(image_path) not in existing_image_paths:
            image_data.append(
                dict(image_path=os.path.basename(image_path), questions=[])
            )
    return image_data

image_data = load_initial_data()
for i, datum in enumerate(image_data):
    if datum['questions']:
        print(i, datum['image_path'], datum['questions'])

@app.post("/api/getData/")
async def get_data():
    #print("fetching data and sending", image_data)
    for datum in image_data:
        if len(datum['questions']) > 0:
            print(datum)
    return JSONResponse(content={"data": image_data})

@app.post("/api/saveData/")
async def save_data(data: ImageData):
    image_data = data.data
    with open(data_path, 'w') as f:
        for image_datum in image_data:
            json.dump(image_datum, f)
            f.write('\n')
    return JSONResponse(content={"message": "Data saved successfully"})

@app.get("/api/getImage/{image_id}")
async def get_image(image_id: int):
    print(f"{image_id=}")
    path = image_data[image_id]['image_path']
    with open(f"{image_dir}/{path}", "rb") as f:
        image_encoded = base64.b64encode(f.read()).decode("utf-8")
    return JSONResponse(content={"image": image_encoded})

