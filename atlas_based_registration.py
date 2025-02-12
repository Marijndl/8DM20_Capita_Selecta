from __future__ import print_function, absolute_import

import os
import warnings
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths
ELASTIX_PATH = r'D:\Elastix\elastix.exe'
TRANSFORMIX_PATH = r'D:\Elastix\transformix.exe'
DATA_PATH = r'D:\capita_selecta\DevelopmentData\DevelopmentData'

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

# Get patient names and select atlas patients
patient_list = [patient for patient in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, patient))]
atlas_patients = patient_list[:5]

from atlas_registration_functions import register_transform

register_patients = [patient for patient in patient_list if patient not in atlas_patients]

# Outer loop: iterate over patients with a progress bar
for patient in tqdm(register_patients[:2], desc="Processing Patients", unit="patient"):
    aggregate_delination = np.empty(1)

    # Inner loop: register each patient to all atlas patients with a progress bar
    for atlas in tqdm(atlas_patients[:4], desc=f"Registering {patient}", unit="atlas", leave=False):
        transformed_delineation_path = register_transform(atlas, patient, DATA_PATH, ELASTIX_PATH, TRANSFORMIX_PATH)
        transformed_delineation = sitk.GetArrayFromImage(sitk.ReadImage(transformed_delineation_path))

        # Add deformed delineation to aggregate
        if aggregate_delination.size == 1:
            aggregate_delination = transformed_delineation
        else:
            aggregate_delination = np.add(aggregate_delination, transformed_delineation)

    # Save the aggregate delineation
    majority_vote = (aggregate_delination >= (len(register_patients) // 2 + 1)).astype(int)
    majority_vote_image = sitk.GetImageFromArray(majority_vote)
    majority_vote_image.SetSpacing([0.488281, 0.488281, 1])  # Each pixel is 0.488281 x 0.488281 x 1 mm^2

    output_dir = r'D:\capita_selecta\results_experiments'
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    output_path = os.path.join(output_dir, f'reg_maj_{patient}.mhd')

    sitk.WriteImage(majority_vote_image, output_path)
    tqdm.write(f"Saved majority vote image for {patient} at {output_path}")
