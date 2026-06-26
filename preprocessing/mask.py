"""
preprocessing/mask.py

Generates per-zone segmentation masks from UWFFA retinal images by detecting
yellow zone boundary lines via HSV thresholding and assigning contours to
anatomical zones 1-10.

Zone layout (right eye / OD):
    9     — Optic disk
    1-4   — Inner quadrants
    5-8   — Outer quadrants
    10    — Outside outermost boundary

Usage:
    python3 mask.py \
        --image_dir /mnt/NAS/Tim/Datasets/Sample_02_25_26_OD \
        --csv_path  /home/tim/uveitis/fold_0/train.csv
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# HSV range for the yellow zone-boundary lines 
YELLOW_LOW  = np.array([20, 100, 100])
YELLOW_HIGH = np.array([35, 255, 255])

MIN_CONTOUR_AREA = 500  #Area threshold for contour (pixels) 


def get_centroid(contour):
    """Returns (cx, cy) of a contour, or (None, None) if area is zero."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def _fill_zone(mask, contour, zone_id):
    cv2.drawContours(mask, [contour], -1, zone_id, thickness=cv2.FILLED)


def _sort_into_grid(contours):
    """
    Sort 4 contours into a 2x2 grid by centroid position.
    Returns (upper_left, upper_right, lower_left, lower_right).
    """
    centroids = [(get_centroid(c), c) for c in contours]
    by_y = sorted(centroids, key=lambda x: x[0][1])
    upper = sorted(by_y[:2], key=lambda x: x[0][0])
    lower = sorted(by_y[2:], key=lambda x: x[0][0])
    return upper, lower


def _assign_outer_ring(mask, outer_contour):
    # Zone 10 is outside the outermost boundary. Fill inside with values and invert
    _fill_zone(mask, outer_contour, 255)
    np.place(mask, mask == 0, 10)
    np.place(mask, mask == 255, 0)


def _assign_outer_quadrants(mask, quad_contours):
    # Zones 5-8: upper-left=5, upper-right=6, lower-right=7, lower-left=8
    upper, lower = _sort_into_grid(quad_contours)
    for zone_id, (_, contour) in zip([5, 6], upper):
        _fill_zone(mask, contour, zone_id)
    for zone_id, (_, contour) in zip([8, 7], lower):
        _fill_zone(mask, contour, zone_id)


def _assign_inner_zones(mask, inner_contours, optic_disk_split):
    # The optic disk is always leftmost. When a quadrant line bisects it,
    # there are 2 disk contours instead of 1.
    centroids = [(get_centroid(c), c) for c in inner_contours]
    by_x = sorted(centroids, key=lambda x: x[0][0])

    n_disk = 2 if optic_disk_split else 1
    for (_, contour) in by_x[:n_disk]:
        _fill_zone(mask, contour, 9)

    # Remaining 4 are inner quadrants: zones 1-4 in same grid order as 5-8
    upper, lower = _sort_into_grid([c for _, c in by_x[n_disk:]])
    for zone_id, (_, contour) in zip([1, 2], upper):
        _fill_zone(mask, contour, zone_id)
    for zone_id, (_, contour) in zip([3, 4], lower):
        _fill_zone(mask, contour, zone_id)


def label_contours(contours, H, W):
    """
    Builds a (H, W) zone mask from a sorted list of 10 or 11 contours.
    """
    assert len(contours) in (10, 11), f"Expected 10 or 11 contours, got {len(contours)}"
    mask = np.zeros((H, W), dtype=np.uint8)

    _assign_outer_ring(mask, contours[0])
    _assign_outer_quadrants(mask, contours[1:5])
    _assign_inner_zones(mask, contours[5:], optic_disk_split=(len(contours) == 11))

    return mask


def find_zone_contours(image):
    """
    Detects yellow zone boundary contours in a UWFFA image.
    Returns contours sorted, or None if count isn't 10 or 11. 
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    # Dilate to close small gaps in the boundary lines
    yellow = cv2.dilate(yellow, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(
        [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA],
        key=cv2.contourArea,
        reverse=True,
    )

    return contours if len(contours) in (10, 11) else None


def create_zone_masks(image_path):
    """
    Generates and saves a zone mask for a single UWFFA image as a .npy file
    Returns the save path, or None if skipped.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Could not read: {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    contours = find_zone_contours(image)
    if contours is None:
        print(f"[Skip] {image_path} — unexpected contour count")
        return None

    labeled_mask = label_contours(contours, H, W)
    save_path = image_path.replace(".png", "_masks.npy")
    np.save(save_path, labeled_mask)
    return save_path


def _normalize_path(raw_path):
    # CSV stores Windows absolute paths (e.g. C:\data\Patient001\scan.png).
    # Strip drive + prefix
    path = raw_path.replace("\\", "/")
    start = path.find("Patient")
    return path[start:] if start != -1 else path


def create_masks_from_csv(image_dir, csv_path):
    df = pd.read_csv(csv_path)
    df["UWFFA"] = df["UWFFA"].dropna().apply(_normalize_path)
    paths = df["UWFFA"].dropna().unique()

    print(f"[count] {len(paths)} unique images")

    ok, skip, err = 0, 0, 0
    for i, img_path in enumerate(paths):
        full_path = str(Path(image_dir) / img_path)
        tag = f"[{i+1}/{len(paths)}]"
        try:
            result = create_zone_masks(full_path)
            if result:
                ok += 1
                print(f"{tag} OK:   {Path(img_path).name}")
            else:
                skip += 1
                print(f"{tag} SKIP: {Path(img_path).name}")
        except Exception as e:
            err += 1
            print(f"{tag} ERR:  {Path(img_path).name} — {e}")

    print(f"\n[done] {ok} saved, {skip} skipped, {err} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute zone masks from a CSV of UWFFA image paths")
    parser.add_argument("--image_dir", required=True, help="Root directory containing images")
    parser.add_argument("--csv_path",  required=True, help="CSV with a 'UWFFA' column of image paths")
    args = parser.parse_args()
    create_masks_from_csv(args.image_dir, args.csv_path)