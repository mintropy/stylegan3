import datetime
import shutil
import json
from typing import Optional, List
import os

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.mount(
    "/api/images", 
    StaticFiles(directory="data/images/"), 
    name="images"
)

@app.get("/api/gen-image/{data_id}/")
def gen_image(data_id: str, count: Optional[int] = 1):
    file_dir = "test.png"
    return FileResponse(file_dir)


@app.get("/api/data-list/")
def read_data_list():
    with open("data/data.json") as f:
        data = json.load(f)
    return data


@app.post("/api/data-list/")
def create_data_list(
    pkl_file: UploadFile, 
    name: Optional[str] = None, 
    img: Optional[UploadFile] = None,
    description: Optional[str] = None,
):
    if (
        pkl_file.content_type != "application/octet-stream"
        or (
            len(pkl_file.filename) >= 4
            and pkl_file.filename[-4:] != '.pkl'
        )
        or (
            img is not None
            and img.content_type[:5] != "image"
        )
    ):
        return Response(status_code=400)
    
    # def file & imgage path
    pkl_file_name = pkl_file.filename
    if name is None:
        name = pkl_file_name
    with open("data/data.json", "r") as f:
        data = json.load(f)
    t = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")
    if os.path.isfile(f"data/{pkl_file_name}"):
        pkl_file_name = f"{pkl_file_name[:-4]}-{t}.pkl"
    with open(f"data/{pkl_file_name}", "wb") as f:
        shutil.copyfileobj(pkl_file.file, f)
    if img is not None:
        img_name = img.filename
        if os.path.isfile(f"data/images/cover/{img_name}"):
            dot_idx = img_name.rindex(".")
            img_name = f"{img_name[:dot_idx]}-{t}{img_name[dot_idx:]}"
        with open(f"data/images/cover/{img_name}", "wb") as image:
            shutil.copyfileobj(img.file, image)
    else:
        img_name = None
    num = 1 if not data else (int(max(data.keys())) + 1)
    data[num] = {
        "pkl": pkl_file_name,
        "name": name,
        "image": f"cover/{img_name}",
        "description": description
    }
    with open("data/data.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


@app.patch("/api/data-list/{data_id}/")
def update_data(
    data_id: str,
    name: Optional[str] = None, 
    img: Optional[UploadFile] = None,
    description: Optional[str] = None,
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
    data_id = str(data_id)
    if data_id not in data:
        return Response(status_code=404)
    t = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")
    if name is not None:
        data[data_id]["name"] = name
    if img is not None:
        prev_img = data[data_id]["image"]
        os.remove(f"data/images/{prev_img}")
        img_name = img.filename
        if os.path.isfile(f"data/images/cover/{img_name}"):
            dot_idx = img_name.rindex(".")
            img_name = f"{img_name[:dot_idx]}-{t}{img_name[dot_idx:]}"
        with open(f"data/images/cover/{img_name}", "wb") as image:
            shutil.copyfileobj(img.file, image)
        data[data_id]["image"] = img_name
    if description is not None:
        data[data_id]["description"] = description
    with open("data/data.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


@app.delete("/api/data-list/{data_id}/")
def update_data(
    data_id: int,
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
    data_id = str(data_id)
    if data_id not in data:
        return Response(status_code=404)
    img, pkl = data[data_id]["image"], data[data_id]["pkl"]
    os.remove(f"data/{pkl}")
    if img is not None:
        os.remove(f"data/images/{img}")
    data.pop(data_id)
    with open("data/data.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


@app.patch("/api/pkl/rename/{data_id}/")
def pkl_rename(
    data_id: str,
    new_name: str
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
    if data_id not in data:
        return Response(status_code=404)
    if new_name[:-4] != ".pkl":
        new_name = f"{new_name}.pkl"
    if os.path.isfile(f"data/{new_name}"):
        return Response(status_code=400)
    old_name = f"data/{data[data_id]['pkl']}"
    os.rename(old_name, f"data/{new_name}")
    data[data_id]["pkl"] = new_name
    with open("data/data.json", "w") as f:
        json.dump(data, f, indent=2)
    return new_name


@app.get("/api/pkl/download/{data_id}/")
def pkl_download(
    data_id: str,
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
    if data_id not in data:
        return Response(status_code=404)
    pkl = data[data_id]["pkl"]
    return FileResponse(f"data/{pkl}")


@app.get("/api/train/image/{data_id}/")
def train_image(
    data_id: str,
):
    file_path = f"./data/images/train/{data_id}"
    if os.path.isdir(file_path):
        file_list = os.listdir(file_path)
    else:
        file_list = []
    return file_list


@app.post("/api/train/image/{data_id}/")
def upload_train_image(
    data_id: str,
    images: List[UploadFile],
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
    if data_id not in data:
        return Response(status_code=404)
    file_path = f"data/images/train/{data_id}"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    for image in images:
        image_path = f"{file_path}/{image.filename}"
        if not os.path.isfile(image_path):
            with open(image_path, "wb") as img:
                shutil.copyfileobj(image.file, img)
    return data_id
