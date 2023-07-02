#!/usr/bin/env python

from __future__ import annotations

import functools

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from model import Model

DESCRIPTION = '# [MobileStyleGAN](https://github.com/bes-dev/MobileStyleGAN.pytorch)'
SAMPLE_IMAGE_DIR = 'https://huggingface.co/spaces/hysts/MobileStyleGAN/resolve/main/samples'
ARTICLE = f'''## Generated images
### FFHQ
- size: 1024x1024
- seed: 0-99
- truncation: 1.0
![FFHQ]({SAMPLE_IMAGE_DIR}/ffhq.jpg)
'''


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, z_dim)).to(device).float()


@torch.inference_mode()
def generate_image(seed: int, truncation_psi: float, generator: str,
                   model: nn.Module, device: torch.device) -> np.ndarray:
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

    z = generate_z(model.mapping_net.style_dim, seed, device)

    out = model(z, truncation_psi=truncation_psi, generator=generator)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


def load_model(device: torch.device) -> nn.Module:
    path = hf_hub_download('public-data/MobileStyleGAN',
                           'models/mobilestylegan_ffhq_v2.pth')
    ckpt = torch.load(path)
    model = Model()
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.mapping_net.style_dim)).to(device)
        model(z)
    return model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(device)

fn = functools.partial(generate_image, model=model, device=device)

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            with gr.Group():
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=0,
                                 randomize=True)
                psi = gr.Slider(label='Truncation psi',
                                minimum=0,
                                maximum=2,
                                step=0.05,
                                value=1.0)
                generator = gr.Radio(label='Generator',
                                     choices=['student', 'teacher'],
                                     type='value',
                                     value='student')
                run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Image(label='Output', type='numpy')
    with gr.Row():
        gr.Markdown(ARTICLE)

    run_button.click(fn=fn, inputs=[seed, psi, generator], outputs=result)
demo.queue(max_size=10).launch()
