"""
E2E Video Colorization Model - Zhang et al. architecture.
Uses pretrained weights from colorizers repository.
"""
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    ReLU,
    Sequential,
    Softmax,
    Upsample,
)

from .color_code import ColorCode

logger = logging.getLogger(__name__)


class RCNN(ColorCode):
    """Recursive Colorization CNN - processes L channel to predict ab channels."""

    def __init__(self, norm_layer=BatchNorm2d) -> None:
        super().__init__()

        model1 = [
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            ReLU(True),
            norm_layer(64),
        ]

        model2 = [
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            ReLU(True),
            norm_layer(128),
        ]

        model3 = [
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            ReLU(True),
            norm_layer(256),
        ]

        model4 = [
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            norm_layer(512),
        ]

        model5 = [
            Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            ReLU(True),
            norm_layer(512),
        ]

        model6 = [
            Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            ReLU(True),
            norm_layer(512),
        ]

        model7 = [
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            norm_layer(512),
        ]

        model8 = [
            ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            ReLU(True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(True),
            Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        ]

        self.model1 = Sequential(*model1)
        self.model2 = Sequential(*model2)
        self.model3 = Sequential(*model3)
        self.model4 = Sequential(*model4)
        self.model5 = Sequential(*model5)
        self.model6 = Sequential(*model6)
        self.model7 = Sequential(*model7)
        self.model8 = Sequential(*model8)

        self.softmax = Softmax(dim=1)
        self.model_out = Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = Upsample(scale_factor=4, mode="bilinear")

    def forward(self, input_l: torch.Tensor) -> torch.Tensor:
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))
        return self.unnormalize_ab(self.upsample4(out_reg))


def load_colorizer(
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    cache_dir: Optional[Path] = None,
) -> RCNN:
    """
    Load the colorization model with optional pretrained weights.

    Args:
        pretrained: Whether to load pretrained weights
        device: Target device (cuda/cpu). Auto-detected if None.
        cache_dir: Directory for caching model weights

    Returns:
        Loaded RCNN model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RCNN().to(device).eval()

    if pretrained:
        url = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"
        try:
            kwargs = {"url": url, "map_location": device, "progress": True}
            if cache_dir is not None:
                kwargs["model_dir"] = str(cache_dir)
            state_dict = torch.hub.load_state_dict_from_url(**kwargs)
            model.load_state_dict(state_dict)
            logger.info("Loaded pretrained colorization weights from %s", url)
        except Exception as e:
            logger.warning("Could not load pretrained weights: %s. Using random init.", e)

    return model
