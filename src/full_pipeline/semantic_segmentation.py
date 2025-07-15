import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


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


def trim_masks(masks, img):
    original_masks = [m["segmentation"].copy() for m in masks]
    trimmed = []
    for i in range(len(masks)):
        seg_i = masks[i]["segmentation"].copy()
        for j in range(i + 1, len(masks)):
            seg_j = original_masks[j]
            seg_i[np.logical_and(seg_i, seg_j)] = False
        area = np.sum(seg_i)
        if area > 0:
            trimmed.append({"segmentation": seg_i, "area": area, "orig_index": i})
            #plot_mask(f"Trimmed mask {i}", seg_i, img)
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
    mask = mask.astype(bool)
    new_mask = mask.copy()
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                continue
            left  = any(mask[y, :x])
            right = any(mask[y, x+1:])
            up    = any(mask[:y, x])
            down  = any(mask[y+1:, x])
            cond1 = left and right and up and down
            cond2 = up and down and (left or right)
            cond3 = (up or down) and left and right
            if cond1 or cond2:
                new_mask[y, x] = True
    return new_mask


def filter_window_masks(masks, filled_facade, gray_img):
    filtered = []
    for m in masks:
        seg = m["segmentation"]
        area = m["area"]
        x, y, w, h = m["bbox"]
        if (filled_facade[seg].sum() / seg.sum()) < 0.95:
            continue
        if area < 200 or area > 10000:
            continue
        if gray_img[y:y+h, x:x+w].mean() >= gray_img.mean() - 25:
            continue
        filtered.append(seg)
    return filtered


def draw_result_image(base, facade_mask, window_masks):
    blended = base.copy()
    blended[facade_mask] = [255, 255, 0]
    for wm in window_masks:
        blended[wm] = [255, 0, 0]

    outline = blended.copy()
    facade_uint8 = facade_mask.astype(np.uint8) * 255
    contours_facade, _ = cv2.findContours(facade_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outline, contours_facade, -1, (0, 0, 0), 2)

    for wm in window_masks:
        wm_uint8 = wm.astype(np.uint8) * 255
        contours_win, _ = cv2.findContours(wm_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outline, contours_win, -1, (0, 0, 255), 1)

    plt.figure(figsize=(10, 7))
    plt.imshow(outline)
    plt.title("Facade & Windows with Contours")
    plt.axis("off")
    plt.tight_layout()
    #plt.show()
    return contours_facade


def export_contours_to_csv(contours, image_name):
    rows = []
    for u, v in [pt[0] for pt in contours[0]]:
        rows.append({"Bildname": image_name, "u": int(u), "v": int(v)})
    df = pd.DataFrame(rows)
    #df.to_csv("C:\\Users\\ledio\\OneDrive\\Studienarbeit\\Code\\full_pipeline_V2\\fassadenkonturen.csv", index=False)

def export_contours_to_df(contours, image_name):
    rows = []
    for u, v in [pt[0] for pt in contours[0]]:
        rows.append({"Bildname": image_name, "u": int(u), "v": int(v)})
    return pd.DataFrame(rows)

def extract_corner_points(contour, epsilon_factor=0.01):
    """Approximiert eine Kontur durch ein Polygon und gibt Eckpunkte zurück."""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [pt[0] for pt in approx]  # Liste von (x, y)

def export_corner_points_to_df(contours, image_name):
    """Wählt größte Kontur, berechnet Ecken und gibt sie als DataFrame zurück."""
    # Robuste Auswahl der größten Kontur
    facade_contour = max(contours, key=cv2.contourArea)
    corner_points = extract_corner_points(facade_contour)

    rows = [{"Bildname": image_name, "u": int(x), "v": int(y)} for (x, y) in corner_points]
    return pd.DataFrame(rows)

def extract_window_corner_points(window_masks, image_name):
    """
            {"Bildname": ..., "fenster_id": 0, "ecken": [(u, v), (u, v), ...]},
            {"Bildname": ..., "fenster_id": 1, "ecken": ...
    """
    window_data = []
    for idx, wm in enumerate(window_masks):
        wm_uint8 = wm.astype(np.uint8) * 255
        contours, _ = cv2.findContours(wm_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        corners = extract_corner_points(contour)
        data = {
            "Bildname": image_name,
            "fenster_id": idx,
            "ecken": [(int(x), int(y)) for (x, y) in corners]
        }
        window_data.append(data)
    return window_data


def process_image(image_path, model_path):
    #IMAGE_PATH = r"d:\DHBW\Studienarbeit\data\20250614_204105.jpg"
    image_name = os.path.basename(image_path)
    #MODEL_PATH = r"C:\Users\ledio\OneDrive\Studienarbeit\Code\SAM\sam_vit_h_4b8939.pth"

    img_bgr = cv2.imread(image_path)
    img_res = cv2.resize(img_bgr, (1024, 768))
    img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=model_path).to(DEVICE)
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=32,
        pred_iou_thresh=0.86, stability_score_thresh=0.92,
        min_mask_region_area=100)

    print("→ Generating masks...")
    masks = gen.generate(img_rgb)
    top10 = sorted(masks, key=lambda m: m["area"], reverse=True)[:10]


    #for i, m in enumerate(top10):
    #    plot_mask(f"Original mask {i}", m["segmentation"], img_rgb) #debug plot

    trimmed = trim_masks(top10, img_rgb)
    trimmed_sorted = sorted(trimmed, key=lambda m: m["area"], reverse=True)
    facade = select_facade(trimmed_sorted, img_rgb.shape[0])

    #plot_mask("Selected Facade", facade, img_rgb) #debug plot

 
    closed = close_mask(facade)
    #plot_mask("Morphological Closing", closed.astype(bool), img_rgb) #debug plot

    refined = largest_connected_component(closed)
    #plot_mask("Largest Connected Component", refined, img_rgb) #debug plot

    filled = fill_between_mask_edges(refined)
    #plot_mask("Filled Facade", filled, img_rgb) #debug plot

    windows = filter_window_masks(masks, filled, gray)
    contours = draw_result_image(img_rgb, filled, windows)

    df_contours = export_contours_to_df(contours, image_name)
    df_corners = export_corner_points_to_df(contours, image_name)

    window_corner_list = extract_window_corner_points(windows, image_name)

    return df_contours, df_corners, window_corner_list

