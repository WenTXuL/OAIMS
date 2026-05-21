import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        target = target.to(dtype=torch.int64)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target_flat = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target_flat).view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()


class CCGLoss(nn.Module):


    def __init__(self, radius=10, num_classes=3):
        super().__init__()
        self.radius = radius
        self.sigma = radius // 3
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def _gaussian_kernel(self):
        coords = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        return torch.exp(-(x ** 2 + y ** 2) / (2 * self.sigma ** 2))

    def forward(self, y_pred, y_true, points, point_labels, device):
        B, L, _ = points.shape
        _, _, H, W = y_true.shape
        y_true = y_true.to(device)
        y_pred = y_pred.softmax(dim=1)

        r = self.radius
        mask = torch.zeros(B, H + 2 * r, W + 2 * r, device=device)

        y_true_onehot = F.one_hot(y_true.long(), num_classes=self.num_classes).squeeze(1).permute(0, 3, 1, 2)

        kernel = self._gaussian_kernel().to(device)

        padded = F.pad(y_true, (r, r, r, r), mode='constant', value=0)
        class_masks = [(padded == c).float() for c in range(self.num_classes)]

        for b in range(B):
            for l in range(L):
                ip = int(points[b, l, 0]) + r
                jp = int(points[b, l, 1]) + r
                c = int(point_labels[b, l, 0])
                region = class_masks[c][b, 0, ip - r:ip + r + 1, jp - r:jp + r + 1]
                mask[b, ip - r:ip + r + 1, jp - r:jp + r + 1] += kernel * region

        mask = mask[:, r:-r, r:-r]
        loss = self.ce_loss(y_true_onehot.float(), y_pred)
        loss = loss * mask
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """Dice-Focal combined loss.

    L_DF = alpha * L_Focal + (1 - alpha) * L_Dice
    """

    def __init__(self, focal_loss, dice_loss, alpha=0.7):
        super().__init__()
        self.focal = focal_loss
        self.dice = dice_loss
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        return self.alpha * self.focal(y_pred, y_true) + (1 - self.alpha) * self.dice(y_pred, y_true)


class CCGDiceFocalLoss(nn.Module):
    """Combined CCG + Dice-Focal loss.

    L_total = beta * L_CCG + gamma * L_DF
    Default: beta=200, gamma=1 as in paper.
    """

    def __init__(self, ccg_loss, df_loss, beta=200, gamma=1):
        super().__init__()
        self.ccg = ccg_loss
        self.df = df_loss
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred, y_true, points, labels, device):
        ccg = self.beta * self.ccg(y_pred, y_true, points, labels, device)
        df = self.gamma * self.df(y_pred, y_true)
        return ccg + df
