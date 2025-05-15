import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# GML file (adjust the path!)
gml_file = r"C:\DHBW\Studienarbeit\Data\Data_LoD2\LoD2_32_533_5279_1_BW_DHBW_ZU.gml"

# XML namespaces
namespaces = {
    'gml': "http://www.opengis.net/gml",
    'bldg': "http://www.opengis.net/citygml/building/1.0",
    'core': "http://www.opengis.net/citygml/1.0"
}

# Parse the GML file 
tree = ET.parse(gml_file)
root = tree.getroot()

# Target building ID (from KIT Model Viewer or the GML file) 
target_id = "DEBW_001000dwAGe"
building = None

for b in root.findall(".//bldg:Building", namespaces):
    gml_id = b.attrib.get('{http://www.opengis.net/gml}id')
    if gml_id == target_id:
        building = b
        break

if building is None:
    print("Building not found!")
    exit()

print(f"Building {target_id} found.")

# Function to extract polygons of a specific surface type
def extract_polygons(surface_type):
    polygons = []
    for surface in building.findall(f".//bldg:{surface_type}", namespaces):
        for poslist in surface.findall(".//gml:posList", namespaces):
            coords = list(map(float, poslist.text.strip().split()))
            if len(coords) % 3 != 0:
                continue  # Skip invalid triplets
            points = [(coords[i], coords[i + 1], coords[i + 2]) for i in range(0, len(coords), 3)]
            polygons.append((surface_type, points))
    return polygons

# Extract ground and wall surfaces 
ground_polys = extract_polygons("GroundSurface")
wall_polys = extract_polygons("WallSurface")
all_polygons = ground_polys + wall_polys

# Prepare 3D plot 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Visualize polygons
for surface_type, poly in all_polygons:
    color = 'green' if surface_type == "GroundSurface" else 'skyblue'
    ax.add_collection3d(Poly3DCollection([poly], alpha=0.5, facecolor=color, edgecolor='k'))

# Automatically scale axes
all_points = [pt for _, poly in all_polygons for pt in poly]
x_vals, y_vals, z_vals = zip(*all_points)
ax.set_xlim(min(x_vals), max(x_vals))
ax.set_ylim(min(y_vals), max(y_vals))
ax.set_zlim(min(z_vals), max(z_vals))

# Function: Set equal axis scaling for realistic proportions
def set_axes_equal(ax):
    """Set equal scaling on all axes for a realistic 3D representation."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)
    x_middle = sum(x_limits) / 2
    y_middle = sum(y_limits) / 2
    z_middle = sum(z_limits) / 2

    ax.set_xlim(x_middle - max_range / 2, x_middle + max_range / 2)
    ax.set_ylim(y_middle - max_range / 2, y_middle + max_range / 2)
    ax.set_zlim(z_middle - max_range / 2, z_middle + max_range / 2)

# Apply equal axis scaling
set_axes_equal(ax)

# Labeling
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of the Building")

# Show the plot
plt.show()
