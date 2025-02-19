import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Pad naar de hoofdmap met de patiëntmappen
base_path = "/Users/joostklis/Desktop/8DM20/DevelopmentData"

def compute_area(mask):
    """Bereken de oppervlakte door het aantal witte pixels te tellen."""
    return np.sum(mask == 255)

def compute_perimeter(mask):
    """Berekent de omtrek (arc length) van het object in het masker."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return cv2.arcLength(max(contours, key=cv2.contourArea), True)

def compute_compactness(mask):
    """Berekent de compactheid van het object in de slice."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    return (perimeter ** 2) / (4 * np.pi * area) if area > 0 else None

def load_mhd_image(file_path):
    """Laad een .mhd masker correct in als een binair masker."""
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image)

# Lijst om resultaten op te slaan
results = []

# Doorloop alle patiëntmappen
for patient_folder in sorted(os.listdir(base_path)):
    patient_path = os.path.join(base_path, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    prostaat_file = None
    mri_file = None

    # Zoek de bestanden in de patiëntmap
    for file in os.listdir(patient_path):
        if file.endswith("prostaat.mhd"):
            prostaat_file = os.path.join(patient_path, file)
        elif file.endswith("mr_bffe.mhd"):
            mri_file = os.path.join(patient_path, file)

    if not prostaat_file or not mri_file:
        continue

    # Laad de 3D maskers
    mask = load_mhd_image(prostaat_file)
    mri = load_mhd_image(mri_file)

    # Verwerk elke slice
    patient_results = []
    for slice_idx in range(mask.shape[0]):  # Loop door elke 2D slice
        slice_mask = (mask[slice_idx] > 0).astype(np.uint8) * 255
        area = compute_area(slice_mask)
        perimeter = compute_perimeter(slice_mask)
        compactness = compute_compactness(slice_mask)
        if compactness is None:
            continue
        patient_results.append([patient_folder, prostaat_file, slice_idx, area, perimeter, compactness])

    results.extend(patient_results)

# Opslaan als CSV
df = pd.DataFrame(results, columns=["Patient", "File", "Slice", "Area", "Perimeter", "Compactness"])
df.to_csv("resultaten_per_slice.csv", index=False)

# Controleer of er gegevens zijn
if df.empty:
    print("Geen gegevens gevonden.")
    exit()

# Zorg dat de DataFrame is gesorteerd
df = df.sort_values(by=["Patient", "Slice"])

# Interactieve plot
def update(val):
    patient_idx = int(patient_slider.val)
    slice_idx = int(slice_slider.val)

    patient_folder = valid_patients[patient_idx]
    patient_results = df[df["Patient"] == patient_folder]

    # Laad de juiste afbeeldingen
    prostaat_file = patient_results.iloc[0]["File"]
    mri_file = prostaat_file.replace("prostaat.mhd", "mr_bffe.mhd")

    mask = load_mhd_image(prostaat_file)
    mri = load_mhd_image(mri_file)

    ax_mask.imshow(mask[slice_idx], cmap='gray')
    ax_mri.imshow(mri[slice_idx], cmap='gray')
    
    ax_mask.set_title(f"Mask - {patient_folder} - Slice {slice_idx}")
    ax_mri.set_title(f"MRI - {patient_folder} - Slice {slice_idx}")

    ax_compactness.clear()
    ax_compactness.plot(patient_results["Slice"], patient_results["Compactness"], label="Compactness")
    
    # Zoek de juiste compactness-waarde
    current_compactness = patient_results[patient_results["Slice"] == slice_idx]["Compactness"].values
    if len(current_compactness) > 0:
        ax_compactness.plot(slice_idx, current_compactness[0], marker="o", markersize=6, color="red")

    ax_compactness.set_xlabel("Slice Index")
    ax_compactness.set_ylabel("Compactness")
    ax_compactness.set_title(f"Compactness for Slice {slice_idx}")
    ax_compactness.grid(True)

    fig.canvas.draw_idle()

# Lijst van unieke patiënten
valid_patients = df["Patient"].unique()

# Setup figuur met 3 kolommen (MRI, Mask, Compactness)
fig, (ax_mri, ax_mask, ax_compactness) = plt.subplots(1, 3, figsize=(15, 5))

# Eerste patiënt en slice
patient_idx = 0
slice_idx = 0
patient_folder = valid_patients[patient_idx]
patient_results = df[df["Patient"] == patient_folder]

# Laad eerste MRI en masker
prostaat_file = patient_results.iloc[0]["File"]
mri_file = prostaat_file.replace("prostaat.mhd", "mr_bffe.mhd")

mask = load_mhd_image(prostaat_file)
mri = load_mhd_image(mri_file)

ax_mri.imshow(mri[slice_idx], cmap='gray')
ax_mask.imshow(mask[slice_idx], cmap='gray')

ax_mri.set_title(f"MRI - {patient_folder} - Slice {slice_idx}")
ax_mask.set_title(f"Mask - {patient_folder} - Slice {slice_idx}")

# Plot compactness
ax_compactness.plot(patient_results["Slice"], patient_results["Compactness"], label="Compactness")

# Zoek correcte compactness voor eerste slice
current_compactness = patient_results[patient_results["Slice"] == slice_idx]["Compactness"].values
if len(current_compactness) > 0:
    ax_compactness.plot(slice_idx, current_compactness[0], marker="o", markersize=6, color="red")

ax_compactness.set_xlabel("Slice Index")
ax_compactness.set_ylabel("Compactness")
ax_compactness.set_title(f"Compactness for Slice {slice_idx}")
ax_compactness.grid(True)

# Sliders toevoegen
patient_slider = Slider(plt.axes([0.1, 0.05, 0.8, 0.03]), 'Patient', 0, len(valid_patients) - 1, valinit=0, valstep=1)
slice_slider = Slider(plt.axes([0.1, 0.01, 0.8, 0.03]), 'Slice', 0, mask.shape[0] - 1, valinit=0, valstep=1)

# Sliders koppelen aan update-functie
patient_slider.on_changed(update)
slice_slider.on_changed(update)

plt.tight_layout()
plt.show()
