import pandas as pd
import numpy as np
import pathlib
import exifread, fractions
from pyproj import Transformer
from numpy.linalg import svd, det, norm
import json


# === EXIF-Hilfsfunktionen ===
def exif_to_deg(values):
    d, m, s = [fractions.Fraction(v.num, v.den) for v in values]
    return float(d + m / 60 + s / 3600)

def get_gps(img_path):
    try:
        with open(img_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
            lat = exif_to_deg(tags["GPS GPSLatitude"].values)
            lon = exif_to_deg(tags["GPS GPSLongitude"].values)
            if tags.get("GPS GPSLatitudeRef", "").values == "S":
                lat = -lat
            if tags.get("GPS GPSLongitudeRef", "").values == "W":
                lon = -lon
            alt = None
            if "GPS GPSAltitude" in tags:
                alt_value = tags["GPS GPSAltitude"].values[0]
                alt = float(alt_value.num) / float(alt_value.den)
                if tags.get("GPS GPSAltitudeRef", None):
                    ref = tags["GPS GPSAltitudeRef"].values[0]
                    if ref == 1:
                        alt = -alt
            return lat, lon, alt
    except:
        pass
    return None, None, None

# === Geo-Konvertierung ===
def correct_height_ellipsoid_to_geoid(h_ellipsoid, correction=53.0):
    if pd.notna(h_ellipsoid):
        return h_ellipsoid - correction
    return pd.NA

def wgs84_to_utm(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    return transformer.transform(lon, lat)

# === Rotationsmatrix aus Richtung berechnen ===
def build_rotation_matrix(d, world_up=np.array([0, 0, 1])):
    z_cam = d / np.linalg.norm(d)
    x_cam = np.cross(world_up, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:
        world_up = np.array([0, 1, 0])
        x_cam = np.cross(world_up, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    R = np.column_stack((x_cam, y_cam, z_cam))
    return R

# === Meshroom-Daten auslesen ===
def load_pose_df(sfm_path):
    with open(sfm_path, "r", encoding="utf-8") as f:
        sfm_data = json.load(f)
    views = {int(v["poseId"]): v for v in sfm_data["views"]}
    poses = {int(p["poseId"]): p for p in sfm_data["poses"]}
    pose_records = []
    for pose_id, pose_data in poses.items():
        view = views.get(pose_id)
        if view is None:
            continue
        image_name = pathlib.Path(view["path"]).name
        transform = pose_data.get("pose", {}).get("transform", {})
        R_raw = transform.get("rotation", [])
        C_raw = transform.get("center", [])
        if len(R_raw) == 9 and len(C_raw) == 3:
            R = np.array([float(r) for r in R_raw]).reshape((3, 3))
            C = np.array([float(c) for c in C_raw])
            fwd = -1 * R @ np.array([0, 0, -1])
            pose_records.append({
                "Bild": image_name,
                "X": C[0], "Y": C[1], "Z": C[2],
                "View_X": fwd[0], "View_Y": fwd[1], "View_Z": fwd[2]
            })
    return pd.DataFrame(pose_records)

# === Hauptfunktion ===
def process_camera_csv(img_root, sfm_path):
    img_root = pathlib.Path(img_root)  # am Anfang der Funktion
    df = load_pose_df(sfm_path)
    for idx, row in df.iterrows():
        lat, lon, alt = get_gps(img_root / row["Bild"])
        df.loc[idx, "Lat"] = lat
        df.loc[idx, "Lon"] = lon
        df.loc[idx, "Height"] = alt
        df.loc[idx, "UTM_Z"] = correct_height_ellipsoid_to_geoid(alt)

    df[["UTM_X", "UTM_Y"]] = df.apply(
        lambda r: wgs84_to_utm(r["Lon"], r["Lat"]) if pd.notna(r["Lat"]) and pd.notna(r["Lon"]) else (pd.NA, pd.NA),
        axis=1, result_type="expand"
    )

    mask = df[["X", "Y", "Z", "UTM_X", "UTM_Y", "UTM_Z"]].notna().all(axis=1)
    src = df.loc[mask, ["X", "Y", "Z"]].to_numpy(float)
    dst = df.loc[mask, ["UTM_X", "UTM_Y", "UTM_Z"]].to_numpy(float)

    if len(src) < 3:
        raise ValueError("At least 3 cameras with GPS required!")

    src_cent = src.mean(axis=0)
    dst_cent = dst.mean(axis=0)
    src_c = src - src_cent
    dst_c = dst - dst_cent
    H = src_c.T @ dst_c
    U, S, Vt = svd(H)
    R = Vt.T @ U.T
    if det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = S.sum() / (src_c ** 2).sum()
    t = dst_cent - scale * R @ src_cent

    print("Scale:", scale)
    print("Rotation:\n", R)
    print("Translation:", t)

    mesh_pos = df[["X", "Y", "Z"]].to_numpy(float)
    utm_pred = (scale * (mesh_pos @ R.T)) + t
    df[["UTM_X_est", "UTM_Y_est", "UTM_Z_est"]] = utm_pred

    mesh_dir = df[["View_X", "View_Y", "View_Z"]].to_numpy(float)
    utm_dir = (mesh_dir @ R.T)
    utm_dir /= np.linalg.norm(utm_dir, axis=1, keepdims=True)
    df[["Dir_E", "Dir_N", "Dir_H"]] = utm_dir

    rot_mats = []
    for d in utm_dir:
        if np.linalg.norm(d) < 1e-6 or np.isnan(d).any():
            rot_mats.append([np.nan]*9)
            continue
        Rmat = build_rotation_matrix(d)
        rot_mats.append(Rmat.flatten().tolist())
    df["RotMat"] = rot_mats

    column_order = [
        "Bild", "X", "Y", "Z",
        "View_X", "View_Y", "View_Z",
        "Lat", "Lon", "Height",
        "UTM_X", "UTM_Y", "UTM_Z",
        "UTM_X_est", "UTM_Y_est", "UTM_Z_est",
        "Dir_E", "Dir_N", "Dir_H", "RotMat"
    ]
    #df[column_order].to_csv(csv_path, index=False)
    print("CSV updated with rotation matrix and transformed data.")
    return df
