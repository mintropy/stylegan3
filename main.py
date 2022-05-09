import datetime
import shutil
import json
from typing import Optional
import os


from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy


app = FastAPI()

app.mount("/api/images", StaticFiles(directory="data/images"), name="images")


@app.get("/api/gen-image/{data_id}/")
def gen_images(data_id: str, count: Optional[int] = 1):
    network_pkl = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl"
    imgs = []

    with open("data/data.json", "r") as f:
        data = json.load(f)
    if data_id != "0" and data_id not in data:
        return Response(status_code=404)

    # input parameter
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    seeds = np.random.randint(2**32 - 1, size=1)
    latent_vector = None
    truncation_psi = 0.95
    noise_mode = 'random'

    # parameter rename
    seeds_ = seeds
    latent_vector = None
    truncation_psi_ = truncation_psi
    noise_mode_ = noise_mode
    # network = network_pkl
    if data_id == "0":
        network = network_pkl
    else:
        network = f"data/{data[data_id]['pkl']}"

    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    label = torch.zeros([1, G.c_dim], device=device)
    # if G.c_dim != 0:
    #     label[:, class_idx] = 1
    class_idx = label

    if latent_vector == None : 
        for seed_idx, seed in enumerate(seeds_):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, class_idx, truncation_psi=truncation_psi_, noise_mode=noise_mode_)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB'))
    else:
        for i in range(len(latent_vector)):
            img = G(latent_vector[i], class_idx, truncation_psi=truncation_psi_, noise_mode=noise_mode_)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB'))
    img = imgs[0]
    img.save(f'data/test.png')
    
    return FileResponse("data/test.png")


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
        if os.path.isfile(f"data/images/{img_name}"):
            dot_idx = img_name.rindex(".")
            img_name = f"{img_name[:dot_idx]}-{t}{img_name[dot_idx:]}"
        with open(f"data/images/{img_name}", "wb") as image:
            shutil.copyfileobj(img.file, image)
    else:
        img_name = None
    num = 1 if not data else (int(max(data.keys())) + 1)
    data[num] = {
        "pkl": pkl_file_name,
        "name": name,
        "image": img_name,
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
    if data_id not in data:
        return Response(status_code=404)
    t = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")
    if name is not None:
        data[data_id]["name"] = name
    if img is not None:
        prev_img = data[data_id]["image"]
        os.remove(f"data/images/{prev_img}")
        img_name = img.filename
        if os.path.isfile(f"data/images/{img_name}"):
            dot_idx = img_name.rindex(".")
            img_name = f"{img_name[:dot_idx]}-{t}{img_name[dot_idx:]}"
        with open(f"data/images/{img_name}", "wb") as image:
            shutil.copyfileobj(img.file, image)
        data[data_id]["image"] = img_name
    if description is not None:
        data[data_id]["description"] = description
    with open("data/data.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


@app.delete("/api/data-list/{data_id}/")
def update_data(
    data_id: str,
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
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
