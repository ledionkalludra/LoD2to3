import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
#from semantic_segmentation import process_image
from camera_direction import process_camera_csv
import pathlib
# Basisverzeichnis
ROOT = pathlib.Path(__file__).resolve().parents[2]
# Bilder zentral verwalten
IMAGE_FILES = [
    "20250614_204823.jpg",
    "20250614_204835.jpg",
    "20250614_204847.jpg",
    "20250614_204858.jpg",
    "20250614_204908.jpg",
    "20250614_204127.jpg",
    "20250614_204329.jpg",
    "20250614_204453.jpg",
    "20250614_204748.jpg",
    "20250614_204804.jpg",
    "20250614_204815.jpg",
    "20250614_204105.jpg",
    "20250614_204923.jpg",
]


# 
IMAGE_INDEX = 12  # change the number of other picture
IMAGE_NAME = IMAGE_FILES[IMAGE_INDEX]
IMAGE_PATH = ROOT / "data/images" / IMAGE_NAME
SFM_PATH = ROOT / "data/meshroom/cameras.sfm"
GML_FILE = ROOT / "data/citygml/LoD2_32_517_5447_1_BW.gml"
MODEL_PATH = ROOT / "models/sam_vit_h_4b8939.pth"
TARGET_IDS = ["DEBW_0010002nb5I", "DEBW_0010002nb5H"]

# not uses
#CSV_CAMFASSADEN_CSV_PATH = ROOT / "data/intermediate/camera.csv"

IMAGE_ID = IMAGE_NAME.split('.')[0]
FASSADEN_CSV_PATH = ROOT / f"output/{IMAGE_ID}_facade_corners.csv"
FENSTER_CSV_PATH = ROOT / f"output/{IMAGE_ID}_window_corners.csv"


TARGET_IDS = ["DEBW_0010002nb5I", "DEBW_0010002nb5H"]
OFFSET_X, OFFSET_Y, OFFSET_Z = 5, 5, 5

#intrinsic camera parameters
f_mm = 5.0
sensorW_mm = 5.8
sensorH_mm = 4.35
img_w, img_h = 1024, 768

#Check if the given image is in the camera direction DataFrame.
def check_image_in_camera_df(image_name, cam_df):

    if image_name in cam_df["Bild"].values:
        print(f"Image '{image_name}' found in camera direction list.")
        return True
    else:
        print(f"Image '{image_name}' NOT found in camera direction list.")
        return False


# Build rotation matrix
def build_rotation_matrix(d, world_up=np.array([0, 0, 1])):
    z_cam = d / np.linalg.norm(d)
    x_cam = np.cross(world_up, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:
        world_up = np.array([0, 1, 0])
        x_cam = np.cross(world_up, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return np.column_stack((x_cam, y_cam, z_cam))

# Extract polygons from GML
def extract_polygons(building, surf_type, ns):
    polys = []
    for surf in building.findall(f".//bldg:{surf_type}", ns):
        for pl in surf.findall(".//gml:posList", ns):
            vals = list(map(float, pl.text.split()))
            pts = [(vals[i], vals[i+1], vals[i+2]) for i in range(0, len(vals), 3)]
            polys.append((surf_type, pts))
    return polys

# Generate camera frustum
def generate_frustum(cam_pos, dir_vec):
    dv = np.array(dir_vec)/np.linalg.norm(dir_vec)
    up = np.array([0,0,1])
    right = np.cross(dv, up)
    if np.linalg.norm(right)<1e-6:
        up = np.array([0,1,0])
        right = np.cross(dv, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, dv); up /= np.linalg.norm(up)
    tn, tv = np.tan(np.arctan(sensorW_mm/(2*f_mm))), np.tan(np.arctan(sensorH_mm/(2*f_mm)))
    def corners(d):
        ctr = cam_pos + dv*d
        h, w = d*tv, d*tn
        return np.array([
            ctr +  h*up +  w*right,
            ctr +  h*up -  w*right,
            ctr -  h*up -  w*right,
            ctr -  h*up +  w*right,
        ])
    return corners(1.0), corners(50.0)

# Find candidate plane
def find_candidate_plane(cam_pos, ray_dir, all_polys, angle_thresh_deg=40, t_min=0.5, t_max=80):
    best_t, best = np.inf, None
    for surf, pts in all_polys:
        if surf != "WallSurface" or len(pts) < 3:
            continue
        p0, p1, p2 = map(np.array, pts[:3])
        normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal /= np.linalg.norm(normal)
        angle = np.degrees(np.arccos(np.clip(np.dot(normal, ray_dir), -1, 1)))
        if angle > angle_thresh_deg:
            continue
        denom = np.dot(normal, ray_dir)
        if abs(denom) < 1e-6:
            continue
        t = np.dot(normal, p0 - cam_pos) / denom
        if t < best_t:
            best_t = t
            best = (p0, normal, pts)
    return best

# Point in polygon 3D
def point_in_polygon_3d(point, polygon_pts, normal):
    polygon_pts = [np.array(p) for p in polygon_pts]
    point = np.array(point)
    normal = normal / np.linalg.norm(normal)
    ref_vec = polygon_pts[1] - polygon_pts[0]
    ref_vec /= np.linalg.norm(ref_vec)
    ortho_vec = np.cross(normal, ref_vec)
    origin = polygon_pts[0]
    proj_2d = []
    for pt in polygon_pts:
        vec = pt - origin
        x = np.dot(vec, ref_vec)
        y = np.dot(vec, ortho_vec)
        proj_2d.append([x, y])
    vec_p = point - origin
    x = np.dot(vec_p, ref_vec)
    y = np.dot(vec_p, ortho_vec)
    path = MplPath(proj_2d)
    return path.contains_point((x, y))

# Find nearest plane by intersection distance
def find_nearest_plane_by_intersection(cam_pos, ray_dir, all_polys):
    min_dist = np.inf
    best_plane = None
    for surf, pts in all_polys:
        if surf != "WallSurface" or len(pts) < 3:
            continue
        p0, p1, p2 = map(np.array, pts[:3])
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal = normal / np.linalg.norm(normal)
        denom = np.dot(normal, ray_dir)
        if abs(denom) < 1e-6:
            continue
        t = np.dot(normal, p0 - cam_pos) / denom
        if t <= 0:
            continue
        hit = cam_pos + t * ray_dir
        if not point_in_polygon_3d(hit, pts, normal):
            continue
        dist = np.linalg.norm(hit - cam_pos)
        if dist < min_dist:
            min_dist = dist
            best_plane = (p0, p1, p2)
    return best_plane

# functions for analysing the hit point with the lod2-points
def match_hit_to_best_polygon(hit: np.ndarray, all_polys: list):
    min_dist = np.inf
    best_match = {}

    for i, (surf_type, pts) in enumerate(all_polys):
        for j, pt in enumerate(pts):
            pt = np.asarray(pt)
            dist = np.linalg.norm(hit - pt)
            if dist < min_dist:
                min_dist = dist
                best_match = {
                    'poly_idx': i,
                    'vertex_idx': j,
                    'dist': dist,
                    'surf_type': surf_type,
                    'vertex': pt
                }
    return best_match

def project_point_to_plane(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Projiziert einen 3D-Punkt orthogonal auf eine Ebene.
    Die Ebene ist definiert durch einen Punkt (plane_point) und Normalenvektor (plane_normal).
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    vec = point - plane_point
    distance = np.dot(vec, plane_normal)
    return point - distance * plane_normal

def compute_plane_basis(plane_normal: np.ndarray):

    # Gibt zwei orthonormale Richtungsvektoren (e1, e2) in der Ebene zurück.

    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    dummy = np.array([0, 0, -1]) if abs(plane_normal[0]) < 0.9 else np.array([0, 1, 0])
    e1 = np.cross(plane_normal, dummy)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(plane_normal, e1)
    return e1, e2

def to_plane_2d_coords(points_3d: np.ndarray, origin_3d: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    #Wandelt 3D-Punkte in 2D-Koordinaten um, bezogen auf (origin_3d, e1, e2).

    rel = points_3d - origin_3d
    u = rel @ e1
    v = rel @ e2
    return np.column_stack((u, v))

    


#def plot_projected_2d(points_a, points_b, label_a="Hit", label_b="LoD2"):

    plt.figure(figsize=(7, 7))
    plt.scatter(points_a[:, 0], points_a[:, 1], c='lime', label=label_a, marker='x')
    plt.scatter(points_b[:, 0], points_b[:, 1], c='orange', label=label_b, edgecolor='k', marker='o')
    for a, b in zip(points_a, points_b):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'gray', alpha=0.5)  # Verbindungslinien
    plt.gca().set_aspect('equal')
    plt.title("2D-Projektion auf Fassadenebene")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from shapely.geometry import Polygon

def evaluate_polygon_overlap(poly1_points, poly2_points):
    """
    Erzeugt zwei Polygone aus Punktlisten und berechnet:
    - Fläche
    - Überlappung
    - IoU (Intersection over Union)
    """
    try:
        poly1 = Polygon(poly1_points).convex_hull
        poly2 = Polygon(poly2_points).convex_hull
    except Exception as e:
        print("Fehler beim Polygonaufbau:", e)
        return None

    if not poly1.is_valid or not poly2.is_valid:
        print("Ungültiges Polygon erkannt.")
        return None

    area1 = poly1.area
    area2 = poly2.area
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union if union > 0 else 0

    return {
        'area_hit': area1,
        'area_lod2': area2,
        'intersection': intersection,
        'union': union,
        'iou': iou
    }

def compute_point_errors(hits_2d, polys_2d):
    """
    Berechnet Fehlermetriken zwischen den Hitpunkten und den LoD2-Zuordnungspunkten (in 2D).
    """
    diffs = hits_2d - polys_2d
    dists = np.linalg.norm(diffs, axis=1)
    return {
        'mean_dist': np.mean(dists),
        'median_dist': np.median(dists),
        'max_dist': np.max(dists),
        'rms_error': np.sqrt(np.mean(dists**2))
    }


# Main function
def main():
    import matplotlib.pyplot as plt

    image_name = os.path.basename(IMAGE_PATH)
    selected_images = [image_name]
    img_root = pathlib.Path(IMAGE_PATH).parent
    #fassaden_df, eckpunkte_df , fenster_ecken_liste = process_image(IMAGE_PATH, MODEL_PATH)
    # Create data from csv
    fassaden_df = pd.read_csv(FASSADEN_CSV_PATH)
    eckpunkte_df = fassaden_df.copy()  
    fenster_df = pd.read_csv(FENSTER_CSV_PATH)

    fenster_ecken_liste = []
    for window_id, group in fenster_df.groupby("fenster_id"):
        coords = group[["u", "v"]].to_numpy().tolist()
        fenster_ecken_liste.append({"fenster_id": window_id, "ecken": coords})

    cam_df = process_camera_csv(img_root, SFM_PATH)

    ns = {"gml":"http://www.opengis.net/gml", "bldg":"http://www.opengis.net/citygml/building/1.0"}
    tree = ET.parse(GML_FILE)
    root = tree.getroot()
    buildings = [b for b in root.findall(".//bldg:Building", ns)
                 if b.get("{http://www.opengis.net/gml}id") in TARGET_IDS]
    all_polys = []
    for b in buildings:
        all_polys += extract_polygons(b, "GroundSurface", ns)
        all_polys += extract_polygons(b, "WallSurface", ns)

    xyz = [pt for _, poly in all_polys for pt in poly]
    xs, ys, zs = zip(*xyz)
    if "UTM_Z_est" not in cam_df or cam_df["UTM_Z_est"].isna().all():
        cam_df["UTM_Z_est"] = np.mean(zs)

    xs += tuple(cam_df["UTM_X_est"])
    ys += tuple(cam_df["UTM_Y_est"])
    zs += tuple(cam_df["UTM_Z_est"])
    x_min, x_max = min(xs)-OFFSET_X, max(xs)+OFFSET_X
    y_min, y_max = min(ys)-OFFSET_Y, max(ys)+OFFSET_Y
    z_min, z_max = min(zs)-OFFSET_Z, max(zs)+OFFSET_Z
    xr, yr, zr = x_max-x_min, y_max-y_min, z_max-z_min

    fx = f_mm / sensorW_mm * img_w
    fy = f_mm / sensorH_mm * img_h
    cx, cy = img_w / 2, img_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    check_image_in_camera_df(image_name, cam_df)

    for idx, row in cam_df.iterrows():
        if row["Bild"] not in selected_images:
            print("1")
            continue
        print("2")
        cam_pos = np.array([row.UTM_X_est, row.UTM_Y_est, row.UTM_Z_est])
        dir_vec = np.array([row.Dir_E, row.Dir_N, row.Dir_H])
        if np.isnan(cam_pos).any() or np.isnan(dir_vec).any():
            continue

        R = np.array(row["RotMat"]).reshape(3, 3)
        ray_center_cam = K_inv @ np.array([cx, cy, 1]) * [-1, -1, 1]
        ray_center_cam /= np.linalg.norm(ray_center_cam)
        ray_center_world = R @ ray_center_cam

        plane = find_nearest_plane_by_intersection(cam_pos, ray_center_world, all_polys)
        if plane is None:
            print("No suitable wall found"); continue

        p0, p1, p2 = plane
        normal = np.cross(p1 - p0, p2 - p0)
        normal /= np.linalg.norm(normal)
        e1, e2 = compute_plane_basis(normal)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for surf, pts in all_polys:
            color = 'lightgrey' if surf == 'GroundSurface' else 'orange'
            ax.add_collection3d(Poly3DCollection([pts], facecolor=color, edgecolor='k', alpha=0.1))
        ax.scatter(*cam_pos, c='red', s=30)

        corner_pixels = eckpunkte_df[eckpunkte_df["Bildname"] == row["Bild"].replace(".jpg", "")][["u", "v"]].to_numpy()

        print("CSV enthält Bildnamen:")
        print(eckpunkte_df["Bildname"].unique())
        print("Aktuell gesucht:", row["Bild"])
        print(corner_pixels)

        hits_projected, polys_projected = [], []

        for (u, v) in corner_pixels:
            ray_cam = K_inv @ np.array([u, v, 1]) * [-1, -1, 1]
            ray_cam /= np.linalg.norm(ray_cam)
            ray_world = R @ ray_cam
            denom = np.dot(normal, ray_world)
            if abs(denom) < 1e-6: continue
            t = np.dot(normal, p0 - cam_pos) / denom
            if t <= 0: continue

            hit = cam_pos + t * ray_world
            match = match_hit_to_best_polygon(hit, all_polys)
            poly_pt = match["vertex"]

            ax.scatter(*hit, color='green')
            ax.scatter(*poly_pt, color='black')
            hits_projected.append(project_point_to_plane(hit, p0, normal))
            polys_projected.append(project_point_to_plane(poly_pt, p0, normal))

        window_hits_all = []
        for win in fenster_ecken_liste:
            fensterpunkte_uv = win["ecken"]
            fenster_hit_pts = []
            for (u, v) in fensterpunkte_uv:
                ray_cam = K_inv @ np.array([u, v, 1]) * [-1, -1, 1]
                ray_cam /= np.linalg.norm(ray_cam)
                ray_world = R @ ray_cam
                denom = np.dot(normal, ray_world)
                if abs(denom) < 1e-6: continue
                t = np.dot(normal, p0 - cam_pos) / denom
                if t <= 0: continue
                hit = cam_pos + t * ray_world
                ax.scatter(*hit, color='darkred', s=2)
                fenster_hit_pts.append(project_point_to_plane(hit, p0, normal))
            window_hits_all.append(np.array(fenster_hit_pts))

        # 3D-Plot
        if len(hits_projected) >= 3:
            hp = np.vstack([hits_projected, hits_projected[0]])
            ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color='green', label='Hit-Fassade')
            print("Hit-Fassade:")
        if len(polys_projected) >= 3:
            lp = np.vstack([polys_projected, polys_projected[0]])
            ax.plot(lp[:, 0], lp[:, 1], lp[:, 2], color='black', label='LoD2-Fassade')
        for fenster in window_hits_all:
            if len(fenster) >= 3:
                fp = np.vstack([fenster, fenster[0]])
                ax.plot(fp[:, 0], fp[:, 1], fp[:, 2], color='red', linewidth=0.5)


        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect((xr, yr, zr))
        ax.legend()
        plt.tight_layout()
        plt.show()

        # 2D-Plot"
        hits_2d  = to_plane_2d_coords(np.array(hits_projected),  p0, e1, e2)
        polys_2d = to_plane_2d_coords(np.array(polys_projected), p0, e1, e2)

        plt.figure(figsize=(8, 7))
        plt.plot(*np.vstack([hits_2d, hits_2d[0]]).T, 'g-', label='Hit-Fassade')
        plt.plot(*np.vstack([polys_2d, polys_2d[0]]).T, 'k-', label='LoD2-Fassade')

        for i, fenster in enumerate(window_hits_all):
            if len(fenster) >= 3:
                fenster2d = to_plane_2d_coords(fenster, p0, e1, e2)
                plt.plot(*np.vstack([fenster2d, fenster2d[0]]).T, 'c-', label='Fenster' if i==0 else None)

        plt.title("2D-Überblick: Hitpunkte, LoD2 und Fenster")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("Testplot")

        result = evaluate_polygon_overlap(hits_2d, polys_2d)
        if result:
            print("\n📐 Polygonvergleich:")
            for key, val in result.items():
                print(f"  {key}: {val:.3f}")

        errors = compute_point_errors(hits_2d, polys_2d)
        print("\n📏 Punktbasierte Abweichungen:")
        for k, v in errors.items():
            print(f"  {k}: {v:.3f} m")

if __name__ == "__main__":
    main()
