import pyvista as pv
import numpy as np
import imageio

# Load segmentation images
fixed_delineation_path = r'D:\capita_selecta\DevelopmentData\DevelopmentData\p109\prostaat.mhd'
moving_delineation_path = r'D:\capita_selecta\DevelopmentData\DevelopmentData\p107\prostaat.mhd'
transformed_delineation_path = r'D:\capita_selecta\results_experiments_copy\result.mhd'

fixed_delineation = imageio.imread(fixed_delineation_path).astype(np.uint8)
moving_delineation = imageio.imread(moving_delineation_path).astype(np.uint8)
transformed_delineation = imageio.imread(transformed_delineation_path).astype(np.uint8)

def get_mesh(volume):
    """Convert binary 3D volume into a PyVista surface mesh using Marching Cubes."""
    grid = pv.wrap(volume)  # Convert to PyVista UniformGrid
    mesh = grid.contour(isosurfaces=[0.5])  # Extract surface where volume > 0
    return mesh

# Generate surface meshes
fixed_mesh = get_mesh(fixed_delineation)
moving_mesh = get_mesh(moving_delineation)
transformed_mesh = get_mesh(transformed_delineation)

# Create plotter
pl = pv.Plotter()

# Add meshes with distinct colors
pl.add_mesh(fixed_mesh, color='red', opacity=0.5)
# pl.add_mesh(moving_mesh, color='blue', opacity=0.5)
pl.add_mesh(transformed_mesh, color='green', opacity=0.5)

# Show the visualization
pl.show()
