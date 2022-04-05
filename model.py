import sys

import torch
import torch.nn as nn

sys.path.insert(0, 'MobileStyleGAN.pytorch')

from core.models.mapping_network import MappingNetwork
from core.models.mobile_synthesis_network import MobileSynthesisNetwork
from core.models.synthesis_network import SynthesisNetwork


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # teacher model
        mapping_net_params = {'style_dim': 512, 'n_layers': 8, 'lr_mlp': 0.01}
        synthesis_net_params = {
            'size': 1024,
            'style_dim': 512,
            'blur_kernel': [1, 3, 3, 1],
            'channels': [512, 512, 512, 512, 512, 256, 128, 64, 32]
        }
        self.mapping_net = MappingNetwork(**mapping_net_params).eval()
        self.synthesis_net = SynthesisNetwork(**synthesis_net_params).eval()
        # student network
        self.student = MobileSynthesisNetwork(
            style_dim=self.mapping_net.style_dim,
            channels=synthesis_net_params['channels'][:-1])

        self.style_mean = nn.Parameter(torch.zeros((1, 512)),
                                       requires_grad=False)

    def forward(self,
                var: torch.Tensor,
                truncation_psi: float = 0.5,
                generator: str = 'student') -> torch.Tensor:
        style = self.mapping_net(var)
        style = self.style_mean + truncation_psi * (style - self.style_mean)
        if generator == 'student':
            img = self.student(style)['img']
        elif generator == 'teacher':
            img = self.synthesis_net(style)['img']
        else:
            raise ValueError
        return img
