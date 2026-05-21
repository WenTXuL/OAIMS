import os
import random

import cv2
import numpy as np
import torch


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def create_file(file):
    if not os.path.exists(file):
        open(file, "w").close()
        return file
    base, ext = os.path.splitext(file)
    counter = 1
    new_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    open(new_path, "w").close()
    print(f"File created: {new_path}")
    return new_path


def train_time(start, end):
    elapsed = end - start
    return int(elapsed / 60), int(elapsed % 60)


def f1_valid_two_classes(y_pred, y_true):
    """F1 score (dice) for the lesion class (class 1) in binary segmentation."""
    smooth = 1e-7
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true != 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return 2 * tp / (2 * tp + fp + fn + smooth)


def pick_rand(label_map, label):
    indices = np.argwhere(label_map == label)
    return indices[random.choice(range(indices.shape[0]))]


def _sample_points_from_error_components(error_map, indices, class_val):

    if len(indices) == 0:
        return []

    for i in range(indices.shape[0]):
        error_map[indices[i, 0], indices[i, 1]] = 1

    total_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
        error_map, 8, cv2.CV_32S,
    )

    results = []
    for comp in range(1, total_labels):
        pt = pick_rand(label_map, comp)
        area = stats[comp, cv2.CC_STAT_AREA]
        results.append([np.array([pt[0], pt[1]]), area, class_val])

    return results


def generate_points(y_true, y_pred, num=1, detach=False):
    """Generate correction clicks for binary segmentation.
    """
    if y_pred.shape[1] != 1:
        y_pred = y_pred.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)

    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.detach().cpu().numpy().astype(int) if detach else y_pred.cpu().numpy().astype(int)

    H, W = y_true.shape[2], y_true.shape[3]
    combined = []

    map_lb = np.zeros((H, W), dtype=np.uint8)
    lb_misclass = np.argwhere((y_true == 1) & (y_pred == 0))[:, 2:]
    combined.extend(_sample_points_from_error_components(map_lb, lb_misclass, 1))

    map_bl = np.zeros((H, W), dtype=np.uint8)
    bl_misclass = np.argwhere((y_true == 0) & (y_pred == 1))[:, 2:]
    combined.extend(_sample_points_from_error_components(map_bl, bl_misclass, 0))

    combined.sort(key=lambda x: x[1])
    return [(x[0][0], x[0][1], x[2]) for x in combined[-num:]]


def generate_points_batch(y_true, y_pred, num=1, detach=False):
    """Generate correction clicks for a batch of images."""
    B = y_true.shape[0]
    points = np.zeros((B, num, 2))
    point_labels = np.zeros((B, num, 1))

    for i in range(B):
        gen_pts = generate_points(y_true[i:i + 1], y_pred[i:i + 1], num, detach=detach)
        for j in range(num):
            if j < len(gen_pts):
                points[i, j, 0] = gen_pts[j][0]
                points[i, j, 1] = gen_pts[j][1]
                point_labels[i, j, 0] = gen_pts[j][2]
            else:
                points[i, j] = points[i, j - 1]
                point_labels[i, j] = point_labels[i, j - 1]

    return points, point_labels
