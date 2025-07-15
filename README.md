# LoD3 Reconstruction with Image Analysis and LoD2 City Models

This project demonstrates how 3D coordinates of facade elements (such as windows) can be reconstructed from semantically segmented images in combination with an existing LoD2 city model. Segmentation is performed automatically using the Segment Anything Model (SAM) and rule-based algorithms. The reconstruction is based on ray projection and intersection with LoD2 facade surfaces.

---

## Visualized Results

Several results of the reconstruction pipeline are visualized as plots:

### 1. Semantically segmented image
- Shows window and facade masks in the original camera image.

### 2. Reconstructed 3D corner points
- Displays all reconstructed 3D corner points of segmented facade und window contours.
- Visualization is performed in world coordinates, based on estimated camera poses and viewing directions.

### 3. Facade points and window points on the LoD2 surface
- Compares reconstructed facade edge points with the LoD2 points.
- All points are projected onto the corresponding LoD2 facade plane.
- Enables geometric plausibility checks and projection accuracy evaluation.

---

## Data Source & License

The LoD2 3D city model used in this project originates from the official open geospatial data provided by:
**Landesamt für Geoinformation und Landentwicklung Baden-Württemberg (LGL)**  
https://www.lgl-bw.de

These open geospatial base data and services are made available free of charge under the terms of the following license:
**Datenlizenz Deutschland – Namensnennung – Version 2.0**  
http://www.govdata.de/dl-de/by-2-0

According to the provider, attribution must be given as follows:
> "Datenquelle: LGL, www.lgl-bw.de, dl-de/by-2-0"

The data has been processed in this project for academic research purposes.

---

## Setup

### external data

The large models and image data are not included in the repository due to storage limitations. Place this files in the correct folder
Down-Load-Link: https://1drv.ms/f/c/9a051f71d26d4954/Ejzjw0HG5ixMuLeYTkTAolABfdNiz-pJ1L-kv7IUlp6g4Q?e=bLSxrv

### installation and paths

This project works out of the box — no manual path adjustments are necessary. All scripts use relative paths, so once the repository is cloned and the dependencies are installed, it is ready to use.

conda env create --name mein_env -f environment.yml
conda activate mein_env


