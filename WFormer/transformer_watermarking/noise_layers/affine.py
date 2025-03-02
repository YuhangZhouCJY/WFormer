from kornia.augmentation import RandomAffine
import torch.nn as nn

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

    def forward(self, noised_and_cover):
        encoded=noised_and_cover[0]
        # encoded1=ToPILImage(encoded)
        # encoded, transform=RandomAffine(degrees=0.0, translate=(0.1, 0.1), scale=(0.7, 0.7), shear=30.0, return_transform=True)(encoded)
        encoded, transform=RandomAffine(degrees=(-15.0, 15.0), translate=(0.1,0.1), scale=(0.69, 0.71), shear=(-0.2,0.2), return_transform=True)(encoded)
        # encoded=AdjustSaturation(saturation_factor=15.0)(encoded)
        # encoded=AdjustBrightness(brightness_factor=self.factor)(encoded)
        # encoded=AdjustContrast(contrast_factor=self.factor)(encoded)
        # encoded=AdjustGamma(gamma=0.9)(encoded)

        return encoded