import os
import pathlib
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --------------------------------------------------
# Pfad- und Bildkonfiguration
# --------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
IMAGE_FILES = [
    "20250614_204823.jpg", "20250614_204835.jpg", "20250614_204847.jpg",
    "20250614_204858.jpg", "20250614_204908.jpg", "20250614_204127.jpg",
    "20250614_204329.jpg", "20250614_204453.jpg", "20250614_204748.jpg",
    "20250614_204804.jpg", "20250614_204815.jpg", "20250614_204105.jpg",
    "20250614_204923.jpg",
]
IMAGE_index = 12  # Index des zu verarbeitenden Bildes
IMAGE_NAME = IMAGE_FILES[IMAGE_index]
IMAGE_PATH = ROOT / "data" / "images" / IMAGE_NAME
MODEL_PATH = ROOT / "models" / "sam_vit_h_4b8939.pth"

# --------------------------------------------------
# Hilfsfunktionen zur Maskenverarbeitung
# --------------------------------------------------
def plot_mask(title, mask, img, color=(255, 255, 0)):
    img_debug = img.copy()
    if mask is not None:
        img_debug[mask] = color
    plt.figure(figsize=(8, 6))
    plt.imshow(img_debug)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ... (restliche Maskenfunktionen unverändert) ...

def trim_masks(masks, img):
    original_masks = [m["segmentation"].copy() for m in masks]
    trimmed = []
    for i, m in enumerate(masks):
        seg_i = m["segmentation"].copy()
        for seg_j in original_masks[i+1:]:
            seg_i[np.logical_and(seg_i, seg_j)] = False
        area = seg_i.sum()
        if area > 0:
            trimmed.append({"segmentation": seg_i, "area": area, "bbox": m.get("bbox", [0,0,seg_i.shape[1], seg_i.shape[0]])})
    return trimmed


def select_facade(trimmed_masks, img_height):
    for m in trimmed_masks:
        mask = m["segmentation"]
        ys, _ = np.nonzero(mask)
        if len(ys) == 0:
            continue
        y_center = np.mean(ys) / img_height
        if 0.3 < y_center < 0.7:
            return mask
    raise RuntimeError("No valid facade mask found")


def close_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)


def largest_connected_component(mask):
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_label = 1 + np.argmax(areas)
        return (lbl == main_label)
    return mask.astype(bool)


def fill_between_mask_edges(mask):
    h, w = mask.shape
    mask_bool = mask.astype(bool)
    new_mask = mask_bool.copy()
    for y in range(h):
        for x in range(w):
            if mask_bool[y, x]:
                continue
            left = mask_bool[y, :x].any()
            right = mask_bool[y, x+1:].any()
            up = mask_bool[:y, x].any()
            down = mask_bool[y+1:, x].any()
            if (left and right and up and down) or (up and down and (left or right)) or ((up or down) and left and right):
                new_mask[y, x] = True
    return new_mask


def filter_window_masks(masks, filled_facade, gray_img):
    filtered = []
    mean_gray = gray_img.mean()
    for m in masks:
        seg = m["segmentation"]
        area = m["area"]
        x, y, w, h = m["bbox"]
        if (filled_facade & seg).sum() / max(seg.sum(),1) < 0.95:
            continue
        if area < 200 or area > 10000:
            continue
        if gray_img[y:y+h, x:x+w].mean() >= mean_gray - 25:
            continue
        filtered.append(seg)
    return filtered


def draw_result_image(img, facade_mask, window_masks, debug=False):
    overlay = img.copy()
    overlay[facade_mask] = [255, 255, 0]
    for wm in window_masks:
        overlay[wm] = [255, 0, 0]
    outline = overlay.copy()
    facade_uint8 = facade_mask.astype(np.uint8) * 255
    contours_facade, _ = cv2.findContours(facade_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outline, contours_facade, -1, (0, 0, 0), 2)
    for wm in window_masks:
        wm_uint8 = wm.astype(np.uint8) * 255
        contours_win, _ = cv2.findContours(wm_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outline, contours_win, -1, (0, 0, 255), 1)
    if debug:
        plt.figure(figsize=(10, 7))
        plt.imshow(outline)
        plt.title("Konturen mit Overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    return outline, contours_facade


def extract_corner_points(contour, epsilon_factor=0.01):
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [tuple(pt[0]) for pt in approx]


def export_facade_corners(contours, image_name):
    facade_contour = max(contours, key=cv2.contourArea)
    corners = extract_corner_points(facade_contour)
    return pd.DataFrame([{"Bildname": image_name, "u": x, "v": y} for x,y in corners])


def export_window_corners(window_masks, image_name):
    rows = []
    for idx, wm in enumerate(window_masks):
        wm_uint8 = wm.astype(np.uint8) * 255
        contours, _ = cv2.findContours(wm_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        main = max(contours, key=cv2.contourArea)
        corners = extract_corner_points(main)
        for x,y in corners:
            rows.append({"Bildname": image_name, "fenster_id": idx, "u": x, "v": y})
    return pd.DataFrame(rows)


def process_image(image_path, model_path, debug=False):
    img_bgr = cv2.imread(str(image_path))
    img_res = cv2.resize(img_bgr, (1024, 768))
    img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=str(model_path)).to(device)
    gen = SamAutomaticMaskGenerator(sam,
        points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92,
        min_mask_region_area=100)

    print(f"→ Generating masks for {image_path.name}...")
    masks = gen.generate(img_rgb)
    top10 = sorted(masks, key=lambda m: m["area"], reverse=True)[:10]

    trimmed = trim_masks(top10, img_rgb)
    facade = select_facade(trimmed, img_rgb.shape[0])
    if debug:
        plot_mask("Rohfassung", facade, img_rgb)

    closed = close_mask(facade)
    refined = largest_connected_component(closed)
    filled = fill_between_mask_edges(refined)
    if debug:
        plot_mask("Verarbeitete Fassade", filled, img_rgb)

    windows = filter_window_masks(masks, filled, gray)
    if debug:
        for i, w in enumerate(windows):
            plot_mask(f"Fenster {i}", w, img_rgb)

    outline, facade_contours = draw_result_image(img_rgb, filled, windows, debug)

    name = IMAGE_NAME.split('.')[0]
    df_fac = export_facade_corners(facade_contours, name)
    df_win = export_window_corners(windows, name)

    if df_win.empty:
        df_win = pd.DataFrame(columns=["Bildname", "fenster_id", "u", "v"])
    fac_csv = ROOT / "output" / f"{name}_facade_corners.csv"
    win_csv = ROOT / "output" / f"{name}_window_corners.csv"
    df_fac.to_csv(str(fac_csv), index=False)
    df_win.to_csv(str(win_csv), index=False)
    print(f"Saved {fac_csv} and {win_csv}")

    plt.figure(figsize=(10, 7))
    plt.imshow(outline)
    plt.scatter(df_fac['u'], df_fac['v'], marker='o', s=50,
                facecolors='none', edgecolors='yellow', label='Fassade Ecken')
    plt.scatter(df_win['u'], df_win['v'], marker='x', s=50, label='Fenster Ecken')
    plt.legend()
    plt.title("Eckpunkte von Fassaden und Fenstern")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Automatischer Aufruf ohne argparse
if __name__ == "__main__":
    process_image(IMAGE_PATH, MODEL_PATH, debug=True)
