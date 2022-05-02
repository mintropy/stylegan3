from fastapi import FastAPI
from fastapi.responses import FileResponse
import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy


app = FastAPI()


@app.get("/api2/gen-image/")
def gen_images():
    network_pkl = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl"
    imgs = []

    # input parameter
    device = torch.device('cuda')
    seeds = np.random.randint(2**32 - 1, size=1)
    latent_vector = None
    truncation_psi = 0.95
    noise_mode = 'random'

    # parameter rename
    seeds_ = seeds
    latent_vector = None
    truncation_psi_ = truncation_psi
    noise_mode_ = noise_mode
    network = network_pkl

    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    print('ㅇㅋ')
    label = torch.zeros([1, G.c_dim], device=device)
    # if G.c_dim != 0:
    #     label[:, class_idx] = 1
    class_idx = label

    if latent_vector == None : 
        print(1)
        for seed_idx, seed in enumerate(seeds_):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, class_idx, truncation_psi=truncation_psi_, noise_mode=noise_mode_)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB'))
    else:
        print(2)
        print(latent_vector)
        for i in range(len(latent_vector)):
            img = G(latent_vector[i], class_idx, truncation_psi=truncation_psi_, noise_mode=noise_mode_)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB'))
    print(imgs)
    img = imgs[0]
    img.save(f'data/test.png')
    
    return FileResponse("data/test.png")
