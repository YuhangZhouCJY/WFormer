import torch.nn as nn
import torch
from kornia.color.adjust import AdjustHue,AdjustSaturation,AdjustContrast,AdjustBrightness,AdjustGamma
import math
from torchvision.transforms import ToPILImage
class Adjust_hue(nn.Module):
    def __init__(self,factor):
        super(Adjust_hue, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=noised_and_cover[0]
        # encoded1=ToPILImage(encoded)
        encoded=AdjustHue(hue_factor=self.factor*math.pi)((encoded))
        # encoded=AdjustSaturation(saturation_factor=15.0)(encoded)
        # encoded=AdjustBrightness(brightness_factor=self.factor)(encoded)
        # encoded=AdjustContrast(contrast_factor=self.factor)(encoded)
        # encoded=AdjustGamma(gamma=0.9)(encoded)

        return encoded
