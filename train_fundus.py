
import argparse
import random
import time
from random import randint

import cv2
import numpy as np
import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_fundus import UNet
from models.losses import FocalLoss, CCGLoss, DiceFocalLoss, CCGDiceFocalLoss
from data.datasets_fundus import get_data
from utils_fundus import (
    seeding, create_dir, create_file, train_time,
    f1_valid_score, generate_points_batch, pick_rand,
)


def gen_components(indices_list, val, val_list=False, min_area=150):
    if not val_list:
        val = [val for _ in range(len(indices_list))]

    components = []
    for val_idx, indices in enumerate(indices_list):
        if len(indices) == 0:
            continue

        H = max(indices[:, 0].max() + 1, 512)
        W = max(indices[:, 1].max() + 1, 512)
        binary_map = np.zeros((H, W), dtype=np.uint8)
        for i in range(indices.shape[0]):
            binary_map[indices[i, 0], indices[i, 1]] = 1

        total_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
            binary_map, 8, cv2.CV_32S
        )

        for comp_label in range(1, total_labels):
            pi, pj = pick_rand(label_map, comp_label)
            area = stats[comp_label, cv2.CC_STAT_AREA]
            components.append([np.array([pi, pj]), area, val[val_idx]])

            remaining_area = area
            while remaining_area > min_area:
                pi, pj = pick_rand(label_map, comp_label)
                remaining_area //= 2
                components.append([np.array([pi, pj]), remaining_area, val[val_idx]])

    components.sort(key=lambda x: x[1], reverse=True)
    return [(x[0][0], x[0][1], x[2]) for x in components]


# ── Training point generation ────────────────────────────────────────────────

def generate_training_points(y_true, image, device, model, num_points):

    blank = torch.zeros(1, 1, image.shape[2], image.shape[3], device=device)
    points, point_labels = generate_points_batch(y_true, blank, num=1)
    point_input = torch.from_numpy(points).to(device, dtype=torch.float)
    label_input = torch.from_numpy(point_labels).to(device, dtype=torch.float)
    image = image.to(device, dtype=torch.float32)

    pred = model(image, point_input, label_input)
    pred = pred.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)

    y_np = y_true.cpu().numpy().astype(int)
    p_np = pred.cpu().numpy().astype(int)

    misclass = {
        0: [
            np.argwhere((y_np == 0) & (p_np == 1))[:, 2:],
            np.argwhere((y_np == 0) & (p_np == 2))[:, 2:],
        ],
        1: [
            np.argwhere((y_np == 1) & (p_np == 0))[:, 2:],
            np.argwhere((y_np == 1) & (p_np == 2))[:, 2:],
        ],
        2: [
            np.argwhere((y_np == 2) & (p_np == 0))[:, 2:],
            np.argwhere((y_np == 2) & (p_np == 1))[:, 2:],
        ],
    }

    combined = []
    for cls_id, error_pairs in misclass.items():
        min_area = 80 if cls_id == 2 else 150
        combined.append(gen_components(error_pairs, cls_id, min_area=min_area))

    type_idx = [0, 0, 0]
    result = []
    for _ in range(num_points):
        cls = random.randint(0, 2)
        if type_idx[cls] >= len(combined[cls]):
            correct_px = np.argwhere(y_np == cls)[:, 2:]
            if correct_px.shape[0] == 0:
                i, j = random.randint(0, 511), random.randint(0, 511)
                result.append((i, j, int(y_np[0, 0, i, j])))
            else:
                pt = correct_px[random.choice(range(correct_px.shape[0]))]
                result.append((pt[0], pt[1], cls))
        else:
            result.append(combined[cls][type_idx[cls]])
            type_idx[cls] += 1

    return result


def generate_training_points_batch(images, y_true, model, device, max_points):

    num_points = random.randint(1, max_points)
    B = y_true.shape[0]
    points = np.zeros((B, num_points, 2))
    point_labels = np.zeros((B, num_points, 1))

    for i in range(B):
        pts = generate_training_points(
            y_true[i:i + 1], images[i:i + 1], device, model, num_points,
        )
        for j in range(num_points):
            points[i, j, 0] = pts[j][0]
            points[i, j, 1] = pts[j][1]
            point_labels[i, j, 0] = pts[j][2]

    return points, point_labels


# ── Training step ────────────────────────────────────────────────────────────

def train_step(images, labels, points, point_labels, model, optimizer, criterion, device):
    images = images.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.float32)
    pts = torch.from_numpy(points).to(device, dtype=torch.float)
    pt_labels = torch.from_numpy(point_labels).to(device, dtype=torch.float)

    outputs = model(images, pts, pt_labels)
    loss = criterion(outputs, labels, pts, pt_labels, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


# ── Validation ───────────────────────────────────────────────────────────────

def validate(model, loader, device, num_points=6):
    
    model.eval()
    val_scores = np.zeros(num_points)
    f1_records = np.zeros((4, num_points))
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            B = labels.size(0)

            blank = torch.zeros(B, 1, images.shape[2], images.shape[3], device=device)
            points, point_labels = generate_points_batch(labels, blank, num=1)

            for i in range(num_points):
                pts = torch.from_numpy(points).to(device, dtype=torch.float32)
                pt_labels = torch.from_numpy(point_labels).to(device, dtype=torch.float32)

                outputs = model(images, pts, pt_labels)
                new_pts, new_pt_labels = generate_points_batch(labels, outputs)
                outputs = outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
                score = f1_valid_score(outputs, labels)

                val_scores[i] += score[1] / 2 + score[2] / 2
                f1_records[:, i] += score

                point_labels = np.concatenate([point_labels, new_pt_labels], axis=1)
                points = np.concatenate([points, new_pts], axis=1)

            total += B

    f1_records /= total
    val_scores /= total
    model.train()
    return val_scores, f1_records


# ── Main training loop ───────────────────────────────────────────────────────

def train(model, train_loader, test_loader, criterion, config, device):
    optimizer = torch.optim.Adam(model.parameters(), config['learning_rate'])

    def lr_schedule(step):
        if step < 45000:
            return 1.0
        elif step < 55000:
            return 0.5
        elif step < 65000:
            return 0.25
        elif step < 75000:
            return 0.1
        return 0.01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    best_val = [0.0]
    batch_ct = 0

    for epoch in tqdm(range(config['epochs'])):
        avg_loss = 0.0
        start = time.time()

        for _, (images, labels) in enumerate(train_loader):
            points, point_labels = generate_training_points_batch(
                images, labels, model, device, config['num_points'],
            )
            loss = train_step(
                images, labels, points, point_labels, model, optimizer, criterion, device,
            )
            scheduler.step()
            avg_loss += loss.item()
            batch_ct += 1

        end = time.time()
        mins, secs = train_time(start, end)
        print(f'Epoch {epoch + 1} | Train time: {mins}m {secs}s')

        if epoch % 3 == 0 and epoch >= 3:
            val_scores, f1_records = validate(model, test_loader, device)

            if val_scores[-1] > best_val[0]:
                print(f'Val improved: {best_val[0]:.4f} -> {val_scores[-1]:.4f}')
                best_val[0] = val_scores[-1]
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, config['best_path'])

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, config['final_path'])


# ── Setup and entry point ────────────────────────────────────────────────────

def build_loss():
    """Build the combined CCG + Dice-Focal loss (Eq. 5)."""
    focal = FocalLoss()
    dice = DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
    df = DiceFocalLoss(focal, dice, alpha=0.7)
    ccg = CCGLoss(radius=10)
    return CCGDiceFocalLoss(ccg, df, beta=200, gamma=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train interactive segmentation model')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--train_img', nargs='+', type=str, required=True,
                        help='Path(s) to training image folder(s)')
    parser.add_argument('--train_mask', nargs='+', type=str, required=True,
                        help='Path(s) to training mask folder(s)')
    parser.add_argument('--val_img', type=str, required=True,
                        help='Path to validation image folder')
    parser.add_argument('--val_mask', type=str, required=True,
                        help='Path to validation mask folder')
    parser.add_argument('--num_points', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    config = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_points=args.num_points,
        seed=randint(1, 3900),
    )
    seeding(config['seed'])

    save_dir = f'{args.save_dir}/seed/{config["seed"]}'
    create_dir(save_dir)
    config['best_path'] = create_file(f'{save_dir}/best.pth')
    config['final_path'] = create_file(f'{save_dir}/final.pth')

    train_sets = []
    for img_dir, mask_dir in zip(args.train_img, args.train_mask):
        train_sets.append(get_data(img_dir, mask_dir, transform='interactive'))
    train_set = torch.utils.data.ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]

    val_set = get_data(args.val_img, args.val_mask)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = UNet(in_c=6, out_c=3).to(device)
    criterion = build_loss()

    train(model, train_loader, test_loader, criterion, config, device)
