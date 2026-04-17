"""
preprocessing/mask.py 

Example Usage
    "python3 mask.py --image_dir /mnt/NAS/Tim/Datasets/Sample_02_25_26_OD --csv_path /home/tim/UVEITIS_OCT_classidication/fold_0/train.csv"

"""
import cv2
import numpy as np
import pandas as pd
import argparse 

from pathlib import Path 


def get_centroid(contour):
    """
    Computes the centroid (center of mass) of a contour using image moments.
    
    Moments are weighted sums of pixel positions:
        m00 = total area (pixel count)
        m10 = sum of all x coordinates
        m01 = sum of all y coordinates
    
    Centroid formula:
        cx = m10 / m00  (mean x position)
        cy = m01 / m00  (mean y position)

    Returns (cx, cy) as ints, or (None, None) if contour has zero area.
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    

def label_contours_for_ten(contours, H, W): 
    labeled_mask = np.zeros((H, W), dtype=np.uint8)


    cv2.drawContours(labeled_mask, [contours[0]], -1, 255, thickness=cv2.FILLED)
    labeled_mask = np.where(labeled_mask == 0, 10, 0).astype(np.uint8) #Label the inversion (Zone 10) of largest contour outer circle

    #Next four largest: Zones 5-8
    #Sort by centroid X and Y coordinate
    quad_contours = contours[1:5] 
    centroids = [(get_centroid(c), c) for c in quad_contours]
    centroids_sorted_by_y = sorted(centroids, key=lambda x: x[0][1])
    upper_two = sorted(centroids_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two = sorted(centroids_sorted_by_y[2:], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([8, 7], lower_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([5, 6], upper_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)

    #5 Contours: Zone 9 (Optic Disk/ Circle) + 4 Inner Quadrants 
    small_contours = contours[5:10]        
    small_centroids = [(get_centroid(c), c) for c in small_contours]
    small_sorted_by_x = sorted(small_centroids, key=lambda x: x[0][0])
    
    leftmost = small_sorted_by_x[0] #Leftmost is optic disk 
    remaining_four = small_sorted_by_x[1:]  
    (_cx, _cy), contour = leftmost               
    cv2.drawContours(labeled_mask, [contour], -1, 9, thickness=cv2.FILLED)

    remaining_sorted_by_y = sorted(remaining_four, key=lambda x: x[0][1])
    upper_two_small = sorted(remaining_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two_small = sorted(remaining_sorted_by_y[2:], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([3, 4], lower_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([1, 2], upper_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)

    return labeled_mask
    

def label_contours_for_eleven(contours, H, W): 
    labeled_mask = np.zeros((H, W), dtype=np.uint8)

    cv2.drawContours(labeled_mask, [contours[0]], -1, 255, thickness=cv2.FILLED) #Placeholder 255 value
    labeled_mask = np.where(labeled_mask == 0, 10, 0).astype(np.uint8)

    quad_contours = contours[1:5] # 
    centroids = [(get_centroid(c), c) for c in quad_contours]
    centroids_sorted_by_y = sorted(centroids, key=lambda x: x[0][1])
    upper_two = sorted(centroids_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two = sorted(centroids_sorted_by_y[2:], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([8, 7], lower_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([5, 6], upper_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)

    small_contours = contours[5:11]
    small_centroids = [(get_centroid(c), c) for c in small_contours]
    small_sorted_by_x = sorted(small_centroids, key=lambda x: x[0][0])
    leftmost_two = small_sorted_by_x[:2]
    remaining_four = small_sorted_by_x[2:]

    for (_cx, _cy), contour in leftmost_two: #Optic disk split into 2 due to quadrant line 
        cv2.drawContours(labeled_mask, [contour], -1, 9, thickness=cv2.FILLED)

    remaining_sorted_by_y = sorted(remaining_four, key=lambda x: x[0][1])
    upper_two_small = sorted(remaining_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two_small = sorted(remaining_sorted_by_y[2:], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([3, 4], lower_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([1, 2], upper_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
        
    return labeled_mask

    
def create_zone_masks(image_path):
    
    #Load Image 
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Could not read image: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    yellow = cv2.dilate(yellow, kernel, iterations=1)

    contours, _ = cv2.findContours(yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 500]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) != 10 and len(contours) != 11:
        print(f"[Skip] {image_path} — found {len(contours)} contours, expected 10/11")
        return

    if len(contours) == 11: 
        labeled_mask = label_contours_for_eleven(contours, H, W) 
    else: 
        labeled_mask = label_contours_for_ten(contours, H, W)

    save_path = image_path.replace(".png", "_masks.npy")
    np.save(save_path, labeled_mask)
    return save_path


def create_masks_from_csv(image_dir, csv_path): 
    df = pd.read_csv(csv_path)
    
    df["Image_File(FA)"] = df["Image_File(FA)"].str.replace("\\", "/", regex=False)
    paths = df["Image_File(FA)"].dropna().unique()
    
    print(f"[count] Processing {len(paths)} unique images...")

    ok, skip, err = 0, 0, 0
    for i, img_path in enumerate(paths):
        full_path = str(Path(image_dir) / img_path)
        try:
            result = create_zone_masks(full_path)
            if result:
                ok += 1
                print(f"[{i+1}/{len(paths)}] OK: {Path(img_path).name}")
            else:
                skip += 1
        except Exception as e:
            err += 1
            print(f"[{i+1}/{len(paths)}] ERROR: {Path(img_path).name} — {e}")

    print(f"\n[Summary] Done — {ok} saved, {skip} skipped, {err} errors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute zone masks from a CSV of image paths")
    parser.add_argument("--image_dir", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with Image_File(FA) column")
    args = parser.parse_args()

    create_masks_from_csv(args.image_dir, args.csv_path)