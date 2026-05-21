import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_guidance_maps(images, points, labels, sigma=5):
    """Generate Gaussian-smoothed guidance maps and concatenate with input images.

    Each click produces a Gaussian blob on the guidance map of its class.
    

    Args:
        images: [B, C, H, W] input images.
        points: [B, N, 2] click coordinates (row, col).
        labels: [B, N, 1] click class labels (0=background, 1=outer-ring, 2=cup).
        sigma: standard deviation for Gaussian smoothing kernel.

    Returns:
        [B, C+3, H, W] concatenated tensor.
    """
    B, _, H, W = images.shape

    cup_signal = torch.zeros(B, H, W, dtype=torch.float32, device=images.device)
    ring_signal = torch.zeros_like(cup_signal)
    bg_signal = torch.zeros_like(cup_signal)
    class_signals = [bg_signal, ring_signal, cup_signal]

    for b in range(B):
        for p in range(points.shape[1]):
            x, y = points[b, p].long()
            label = int(labels[b, p, 0])
            if 0 <= x < H and 0 <= y < W:
                class_signals[label][b, x, y] = 1

    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=images.device) - kernel_size // 2
    gx, gy = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    smoothed = []
    for signal in [cup_signal, ring_signal, bg_signal]:
        s = F.conv2d(signal.unsqueeze(1), kernel, padding="same")
        s_min = torch.amin(s, dim=(2, 3), keepdim=True)
        s_max = torch.amax(s, dim=(2, 3), keepdim=True)
        s = (s - s_min) / (s_max - s_min + 1e-8)
        smoothed.append(s)

    return torch.cat([images] + smoothed, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_c, affine=False)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_c, affine=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        return features, self.pool(features)


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with click guidance maps as additional input channels.

    Input: 3-channel image + 3 Gaussian guidance maps (cup, outer-ring, background)
    Output: 3-class segmentation map
    """

    def __init__(self, in_c=6, out_c=3):
        super().__init__()
        self.e1 = EncoderBlock(in_c, 12)
        self.e2 = EncoderBlock(12, 72)
        self.e3 = EncoderBlock(72, 144)
        self.e4 = EncoderBlock(144, 288)

        self.b = ConvBlock(288, 576)

        self.d1 = DecoderBlock(576, 288)
        self.d2 = DecoderBlock(288, 144)
        self.d3 = DecoderBlock(144, 72)
        self.d4 = DecoderBlock(72, 12)

        self.outputs = nn.Conv2d(12, out_c, kernel_size=1)

    def forward(self, images, points, labels, train_attention=True):
        x = generate_guidance_maps(images, points, labels, sigma=5)

        s1, x = self.e1(x)
        s2, x = self.e2(x)
        s3, x = self.e3(x)
        s4, x = self.e4(x)

        x = self.b(x)

        x = self.d1(x, s4)
        x = self.d2(x, s3)
        x = self.d3(x, s2)
        x = self.d4(x, s1)

        return self.outputs(x)
