import datetime
import shutil
import io
import json
from typing import Optional, List
import os
import zipfile


from fastapi import FastAPI, UploadFile, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import PIL.Image
import torch


import dnnlib
import legacy


# set maximum image generate
MAXIMUM_IMAGE_GENERATE = 2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/api/images", 
    StaticFiles(directory="data/images/"), 
    name="images"
)


@app.get(
    "/api/gen-image/{data_id}/",
    tags=["images"],
)
async def gen_images(data_id: str, count: Optional[int] = 1):

    with open("data/data.json", "r") as f:
        data = json.load(f)
    if data_id != "0" and data_id not in data:
        return Response(status_code=404)

    if count > MAXIMUM_IMAGE_GENERATE:
        count = MAXIMUM_IMAGE_GENERATE
    # input parameter
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    latent_vector = None
    truncation_psi = 0.95
    noise_mode = 'random'

    # parameter rename
    latent_vector = None
    truncation_psi_ = truncation_psi
    noise_mode_ = noise_mode
    # network = network_pkl
    network = f"data/{data[data_id]['pkl']}"

    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    label = torch.zeros([1, G.c_dim], device=device)
    # if G.c_dim != 0:
    #     label[:, class_idx] = 1
    class_idx = label

    if os.path.isdir("data/images/gen"):
        shutil.rmtree("data/images/gen")
    os.mkdir("data/images/gen")
    imgs = []
    imgs_path = []
    for i in range(count):
        seeds = np.random.randint(2**32 - 1, size=1)
        seeds_ = seeds
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
        img = imgs[-1]
        img.save(f'data/images/gen/{data_id}_{i}.png')
        imgs_path.append(f'data/images/gen/{data_id}_{i}.png')
    
    if count == 1:
        return FileResponse(f"data/images/gen/{data_id}_0.png")
    zip_filename = "archive.zip"
    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for fpath in imgs_path:
        # Calculate path for file in zip
        fdir, fname = os.path.split(fpath)
        # Add file, at correct path
        zf.write(fpath, fname)
    
    # Must close zip for all contents to be written
    zf.close()
    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(
        s.getvalue(), 
        media_type="application/x-zip-compressed", 
        headers=
            {
            'Content-Disposition': f'attachment;filename={zip_filename}'
            }
    )
    return resp


@app.get(
    "/api/data-list/",
    tags=["data list"],
)
async def read_data_list():
    with open("data/data.json") as f:
        data = json.load(f)
    return data


@app.post(
    "/api/data-list/",
    tags=["data list"],
)
async def create_data_list(
    pkl_file: UploadFile, 
    name: Optional[str] = None, 
    img: Optional[str] = None,
    description: Optional[str] = None,
    fid: Optional[float] = Query(default=None, gt=0),
    kimg: Optional[int] = Query(default=None, gt=0),
):
    if (
        pkl_file.content_type != "application/octet-stream"
        or (
            len(pkl_file.filename) >= 4
            and pkl_file.filename[-4:] != '.pkl'
        )
    ):
        return Response(status_code=400)
    
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
    num = 1 if not data else (int(max(data.keys())) + 1)
    data[num] = {
        "pkl": pkl_file_name,
        "name": name,
        "image": img,
        "description": description,
        "fid": fid,
        "kimg": kimg,
    }
    with open("data/data.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


@app.patch(
    "/api/data-list/{data_id}/",
    tags=["data list"],
)
async def update_data(
    data_id: str,
    name: Optional[str] = None, 
    img: Optional[str] = None,
    description: Optional[str] = None,
    fid: Optional[float] = Query(default=None, gt=0),
    kimg: Optional[int] = Query(default=None, gt=0),
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
    data_id = str(data_id)
    if data_id not in data:
        return Response(status_code=404)
    if name is not None:
        data[data_id]["name"] = name
    if img is not None:
        data[data_id]["image"] = img
    if description is not None:
        data[data_id]["description"] = description
    if fid is not None:
        data[data_id]["fid"] = fid
    if kimg is not None:
        data[data_id]["kimg"] = kimg
    with open("data/data.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


@app.delete(
    "/api/data-list/{data_id}/",
    tags=["data list"],
)
async def delete_data(
    data_id: int,
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
    data_id = str(data_id)
    if data_id not in data:
        return Response(status_code=404)
    pkl = data[data_id]["pkl"]
    os.remove(f"data/{pkl}")
    data.pop(data_id)
    with open("data/data.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


@app.patch(
    "/api/pkl/rename/{data_id}/",
    tags=["pkl"],
)
async def pkl_rename(
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


@app.get(
    "/api/pkl/download/{data_id}/",
    tags=["pkl"],
)
async def pkl_download(
    data_id: str,
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
    if data_id not in data:
        return Response(status_code=404)
    pkl = data[data_id]["pkl"]
    return FileResponse(f"data/{pkl}")


@app.get(
    "/api/train/image/{data_id}/",
    tags=["train"],
)
async def train_image(
    data_id: str,
):
    file_path = f"./data/images/train/{data_id}"
    if os.path.isdir(file_path):
        file_list = os.listdir(file_path)
    else:
        file_list = []
    return file_list


@app.post(
    "/api/train/image/{data_id}/",
    tags=["train"],
)
async def upload_train_image(
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


@app.delete(
    "/api/train/image/{data_id}/",
    tags=["train"],
)
async def upload_train_image(
    data_id: str,
):
    with open("data/data.json", "r") as f:
        data = json.load(f)
    if data_id not in data:
        return Response(status_code=404)
    file_path = f"data/images/train/{data_id}"
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    return Response(status_code=204)
