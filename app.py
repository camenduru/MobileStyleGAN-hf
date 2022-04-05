#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from model import Model

ORIGINAL_REPO_URL = 'https://github.com/bes-dev/MobileStyleGAN.pytorch'
TITLE = 'bes-dev/MobileStyleGAN.pytorch'
DESCRIPTION = f'This is a demo for {ORIGINAL_REPO_URL}.'
SAMPLE_IMAGE_DIR = 'https://huggingface.co/spaces/hysts/MobileStyleGAN/resolve/main/samples'
ARTICLE = f'''## Generated images
### FFHQ
- size: 1024x1024
- seed: 0-99
- truncation: 1.0
![FFHQ]({SAMPLE_IMAGE_DIR}/ffhq.jpg)
'''

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


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
    path = hf_hub_download('hysts/MobileStyleGAN',
                           'models/mobilestylegan_ffhq_v2.pth',
                           use_auth_token=TOKEN)
    ckpt = torch.load(path)
    model = Model()
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.mapping_net.style_dim)).to(device)
        model(z)
    return model


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    model = load_model(device)

    func = functools.partial(generate_image, model=model, device=device)
    func = functools.update_wrapper(func, generate_image)

    gr.Interface(
        func,
        [
            gr.inputs.Number(default=0, label='Seed'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=1.0, label='Truncation psi'),
            gr.inputs.Radio(['student', 'teacher'],
                            type='value',
                            default='student',
                            label='Generator'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
