from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk

def registrate_atlas_patient(atlas, patient, DATA_PATH, OUTPUT_PATH, ELASTIX_PATH, TRANSFORMIX_PATH):

    OUTPUT_DIR = os.path.join(OUTPUT_PATH, fr'reg_{patient}_{atlas}')
    # Make a results directory if non exists
    if os.path.exists(OUTPUT_DIR) is False:
        os.mkdir(OUTPUT_DIR)

    # Define the paths to the two images
    fixed_image_path = os.path.join(DATA_PATH, patient, 'mr_bffe.mhd')
    moving_image_path = os.path.join(DATA_PATH, atlas, 'mr_bffe.mhd')

    # Define a new elastix object 'el' with the correct path to elastix
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

    # Register the Atlas to the patients image
    el.register(
        fixed_image=fixed_image_path,
        moving_image=moving_image_path,
        parameters=[os.path.join(DATA_PATH, 'parameterswithpenalty.txt')],
        output_dir=OUTPUT_DIR,
        verbose=False)

    return None

def combine_atlas_registrations(atlas_patients, register_patients, OUTPUT_PATH, DATA_PATH, TRANSFORMIX_PATH):

    for patient in register_patients:
        aggregate_delination = np.empty(1)
        flag = False

        for atlas in atlas_patients:
            OUTPUT_DIR = os.path.join(OUTPUT_PATH, fr'reg_{patient}_{atlas}')
            
            if not os.path.exists(OUTPUT_DIR):
                flag = True
                break

            # Make a new transformix object
            transform_path = os.path.join(OUTPUT_DIR, 'TransformParameters.0.txt')
            tr = elastix.TransformixInterface(parameters=transform_path,
                                              transformix_path=TRANSFORMIX_PATH)

            # Transform the atlas's delineation with the found transformation parameters
            moving_delineation_path = os.path.join(DATA_PATH, atlas, 'prostaat.mhd')
            transformed_delineation_path = tr.transform_image(moving_delineation_path, output_dir=OUTPUT_DIR,
                                                              verbose=False)

            transformed_delineation = sitk.GetArrayFromImage(sitk.ReadImage(transformed_delineation_path))

            # Add deformed delineation to aggregate
            if aggregate_delination.size == 1:
                aggregate_delination = transformed_delineation
            else:
                aggregate_delination = np.add(aggregate_delination, transformed_delineation)
        if flag:
            print(f"Failed to make composite for patient: {patient}")
            flag = False
            continue

        # Save the aggregate delineation
        majority_vote = (aggregate_delination >= (len(atlas_patients) // 2 + 1)).astype(int)
        majority_vote_image = sitk.GetImageFromArray(majority_vote)
        majority_vote_image.SetSpacing([0.488281, 0.488281, 1])  # Each pixel is 0.488281 x 0.488281 x 1 mm^2

        os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists
        output_path = os.path.join(OUTPUT_DIR, f'reg_maj_{patient}.mhd')

        sitk.WriteImage(majority_vote_image, output_path)

    return None
