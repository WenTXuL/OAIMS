
import argparse
import gc

import numpy as np
import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_brain import UNet
from models.losses import FocalLoss, CCGLoss, DiceFocalLoss, CCGDiceFocalLoss
from data.datasets_brain import get_brain_data
from utils_brain import f1_valid_two_classes, generate_points_batch


def load_model(model_path, device):
    """load pretrained weights."""
    model = UNet(in_c=3, out_c=2)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(device)


def build_loss():
    """Initialize CCG + Dice-Focal loss """
    focal = FocalLoss()
    dice = DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
    df = DiceFocalLoss(focal, dice, alpha=0.7)
    ccg = CCGLoss(radius=10, num_classes=2)
    return CCGDiceFocalLoss(ccg, df, beta=200, gamma=1)


def run_interactive_inference(model, images, labels, device, num_points,
                              optimizer=None, criterion=None):
    """Simulate T interactive clicks and record metrics at each step.

    When optimizer and criterion are provided, Mid-Interaction (MI) adaptation
    is performed.

    Returns:
        p_final: Prediction after all clicks [B, 1, H, W]
        first_click: (point_input, point_label_input) for the first click
        f1_scores: per-click F1 scores [T]
    """
    B, _, H, W = images.shape
    blank = torch.zeros(B, 1, H, W, device=device)
    points, point_labels = generate_points_batch(labels, blank, num=1)

    f1_scores = np.zeros(num_points)
    first_click = None

    for t in range(num_points):
        pts = torch.from_numpy(points).to(device, dtype=torch.float32)
        pt_labels = torch.from_numpy(point_labels).to(device, dtype=torch.float32)

        logits = model(images, pts, pt_labels, train_attention=True)
        pred = logits.detach().softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
        new_pts, new_pt_labels = generate_points_batch(labels, pred, 1)

        if t == 0:
            first_click = (pts.clone(), pt_labels.clone())

        f1_scores[t] = f1_valid_two_classes(pred, labels)

        point_labels = np.concatenate([point_labels, new_pt_labels], axis=1)
        points = np.concatenate([points, new_pts], axis=1)

        # Mid-Interaction: fine-tune between clicks (skip the last click)
        if optimizer is not None and criterion is not None and t != num_points - 1:
            with torch.no_grad():
                next_pts = torch.from_numpy(points).to(device, dtype=torch.float32)
                next_pt_labels = torch.from_numpy(point_labels).to(device, dtype=torch.float32)
                next_logits = model(images, next_pts, next_pt_labels, train_attention=True)
                next_pred = next_logits.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)

            mi_pts = torch.from_numpy(new_pts).to(device, dtype=torch.float32)
            mi_labels = torch.from_numpy(new_pt_labels).to(device, dtype=torch.float32)
            loss = criterion(logits, next_pred.float(), mi_pts, mi_labels, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return pred, first_click, f1_scores


def post_interaction_adapt(model, optimizer, criterion, images, p_final,
                           first_click, device, num_correction_clicks=10):
    """Post-Interaction adaptation: Stage 1 + Stage 2.

    Stage 1: Fine-tune with the localization click (first click).
    Stage 2: Generate correction clicks from Stage-1 errors, fine-tune again.
    """
    model.train()
    p_final_float = p_final.to(device, dtype=torch.float32)
    first_pts, first_labels = first_click

    # Stage 1: localization click
    stage1_output = model(images, first_pts, first_labels, train_attention=True)
    loss = criterion(stage1_output, p_final_float, first_pts, first_labels, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Stage 2: correction clicks 
    stage1_pred = stage1_output.detach().softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
    corr_pts, corr_labels = generate_points_batch(
        p_final_float, stage1_pred, num=num_correction_clicks,
    )
    corr_pts_t = torch.from_numpy(corr_pts).to(device, dtype=torch.float)
    corr_labels_t = torch.from_numpy(corr_labels).to(device, dtype=torch.float)

    stage2_output = model(images, corr_pts_t, corr_labels_t, train_attention=True)
    loss = criterion(stage2_output, p_final_float, corr_pts_t, corr_labels_t, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run_adaptation(model, loader, optimizer, criterion, device, num_points,
                   dataset_name, enable_mi=True):
    """Run online adaptation on a full test sequence."""
    f1_totals = np.zeros(num_points)
    total = 0

    mi_optimizer = optimizer if enable_mi else None
    mi_criterion = criterion if enable_mi else None

    for idx, (images, labels, pathimg, _) in enumerate(tqdm(loader, desc=dataset_name)):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device)

        p_final, first_click, f1_scores = \
            run_interactive_inference(model, images, labels, device, num_points,
                                     optimizer=mi_optimizer, criterion=mi_criterion)

        f1_totals += f1_scores
        total += 1

        post_interaction_adapt(model, optimizer, criterion, images, p_final,
                               first_click, device)

        del images, labels, p_final, first_click, f1_scores
        gc.collect()
        torch.cuda.empty_cache()

    f1_avg = (f1_totals / total).round(4)

    print(f'\n{dataset_name} ({total} images)')
    print(f'  Lesion F1: {_fmt(f1_avg)}')


def _fmt(arr):
    return ', '.join(f'{v:.4f}' for v in arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Brain online adaptation (MI + PI)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_points', type=int, default=10)
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to folder containing test images')
    parser.add_argument('--mask_path', type=str, required=True,
                        help='Path to folder containing test masks')
    parser.add_argument('--disable_mi', action='store_true',
                        help='Disable Mid-Interaction adaptation (PI only)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    model = load_model(args.model_path, device)
    dataset = get_brain_data(args.img_path, args.mask_path, return_path=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = build_loss()

    run_adaptation(
        model, loader, optimizer, criterion, device,
        args.num_points, 'brain',
        enable_mi=not args.disable_mi,
    )
