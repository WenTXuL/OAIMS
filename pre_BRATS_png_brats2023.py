import argparse
import os
import numpy as np
import SimpleITK as sitk
import cv2
import cc3d
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def preprocess(name, img_dir, gt_dir, out_img, out_mask, prefix,
               img_suffix, gt_suffix):
    image_name = name.split(gt_suffix)[0] + img_suffix
    gt = np.uint8(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_dir, name))))

    for s in range(gt.shape[0]):
        gt[s] = cc3d.dust(gt[s], threshold=10, connectivity=8, in_place=True)

    z_index = np.unique(np.where(gt > 0)[0])
    if len(z_index) == 0:
        return

    gt_roi = np.uint8(gt[z_index])
    gt_roi[gt_roi > 0] = 255

    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(img_dir, image_name))).astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    bg_mask = img == 0
    lo, hi = np.percentile(img[img > 0], 0.5), np.percentile(img[img > 0], 99.5)
    img = np.clip(img, lo, hi)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img[bg_mask] = 0
    img_roi = np.uint8(img[z_index])

    max_idx = np.argmax(np.sum(gt_roi == 255, axis=(1, 2)))
    stem = name.split(gt_suffix)[0]
    tag = f"{prefix}{stem}-{str(max_idx).zfill(3)}"
    cv2.imwrite(os.path.join(out_img, f"{tag}_img.png"),
                cv2.resize(img_roi[max_idx], (256, 256), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(os.path.join(out_mask, f"{tag}_gt.png"),
                cv2.resize(gt_roi[max_idx], (256, 256), interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--modality", type=str, default="T1c")
    parser.add_argument("--img_suffix", type=str, default="t1c.nii.gz")
    parser.add_argument("--gt_suffix", type=str, default="seg.nii.gz")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    prefix = f"{args.modality}_Tumor_"
    out_img = os.path.join(args.output_path, "image")
    out_mask = os.path.join(args.output_path, "mask")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    names = sorted([n for n in os.listdir(args.gt_path)
                    if os.path.exists(os.path.join(args.img_path,
                       n.split(args.gt_suffix)[0] + args.img_suffix))])
    print(f"Found {len(names)} volumes")

    fn = partial(preprocess, img_dir=args.img_path, gt_dir=args.gt_path,
                 out_img=out_img, out_mask=out_mask, prefix=prefix,
                 img_suffix=args.img_suffix, gt_suffix=args.gt_suffix)
    with Pool(args.num_workers) as p:
        for _ in tqdm(p.imap_unordered(fn, names), total=len(names)):
            pass
