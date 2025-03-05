import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure

def load_mhd_image(file_path):
    """Laadt een .mhd masker als een numpy array."""
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image)

def calculate_svr(binary_mask):
    """Bereken Surface-to-Volume Ratio (SVR)."""
    verts, faces, _, _ = measure.marching_cubes(binary_mask.astype(np.uint8), level=0)
    S = measure.mesh_surface_area(verts, faces)  # Oppervlakte
    V = np.sum(binary_mask)  # Volume = aantal voxels
    return S / V  # Surface-to-Volume Ratio (SVR)

base_path = "/Users/joostklis/Desktop/8DM20/DevelopmentData"
svr_values = []
patient_names = []

for patient_folder in sorted(os.listdir(base_path)):
    patient_path = os.path.join(base_path, patient_folder)
    if not os.path.isdir(patient_path):
        continue
    prostaat_file = os.path.join(patient_path, 'prostaat.mhd')
    if not os.path.exists(prostaat_file):
        continue
    
    mask_3d = load_mhd_image(prostaat_file)
    binary_mask = mask_3d > 0  # Zet om naar een binaire vorm (True/False)
    svr = calculate_svr(binary_mask)
    
    svr_values.append(svr)
    patient_names.append(patient_folder)
    
    print(f"Patiënt: {patient_folder}, Surface-to-Volume Ratio: {svr:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(patient_names, svr_values, color='salmon')
plt.xlabel('Patiënt')
plt.ylabel('Surface-to-Volume Ratio (SVR)')
plt.title('Surface-to-Volume Ratio per Patiënt')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
