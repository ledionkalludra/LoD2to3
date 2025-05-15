# LoD2to3

This project visualizes a building from a CityGML (.gml) file in 3D using Python. It extracts **WallSurface** and **GroundSurface** polygons of a specific building based on its GML ID and visualizes them using `matplotlib`.

## 🔍 Features

- Parsing CityGML data using `xml.etree.ElementTree`
- Extracting building polygons (Wall/Ground surfaces)
- 3D rendering using `matplotlib`
- Automatic axis scaling and equal aspect ratio for accurate geometry display

You need to adjust the path to your GML file in the code:
```python
gml_file = r"C:\DHBW\Studienarbeit\Data\Data_LoD2\LoD2_32_533_5279_1_BW_DHBW_ZU.gml"
