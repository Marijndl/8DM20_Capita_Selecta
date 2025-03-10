from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
import seaborn as sns

def registrate_atlas_patient(atlas, patient, DATA_PATH, OUTPUT_PATH, ELASTIX_PATH, verbose):

    OUTPUT_DIR = os.path.join(OUTPUT_PATH, fr'reg_{patient}_{atlas}')
    PARAM_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "parameter_files")
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
        parameters=[
            os.path.join(PARAM_DIR, 'Par0001translation.txt'),
            os.path.join(PARAM_DIR, 'Par0001bspline16.txt'),
                    ],
        output_dir=OUTPUT_DIR,
        verbose=verbose)

    return None

def combine_atlas_registrations(atlas_patients, register_patients, OUTPUT_PATH, DATA_PATH, TRANSFORMIX_PATH, verbose=False):

    for patient in register_patients:
        print(f"Patient: {patient}")
        aggregate_delination = np.empty(1)
        flag = False

        for atlas in atlas_patients:
            OUTPUT_DIR = os.path.join(OUTPUT_PATH, fr'reg_{patient}_{atlas}')
            # Make a new transformix object
            transform_path = os.path.join(OUTPUT_DIR, 'TransformParameters.0.txt')
            
            if not os.path.exists(transform_path):
                continue

            tr = elastix.TransformixInterface(parameters=transform_path,
                                              transformix_path=TRANSFORMIX_PATH)

            # Transform the atlas's delineation with the found transformation parameters
            moving_delineation_path = os.path.join(DATA_PATH, atlas, 'prostaat.mhd')
            transformed_delineation_path = tr.transform_image(moving_delineation_path, output_dir=OUTPUT_DIR,
                                                              verbose=verbose)

            transformed_delineation = sitk.GetArrayFromImage(sitk.ReadImage(transformed_delineation_path))

            # Add deformed delineation to aggregate
            if aggregate_delination.size == 1:
                aggregate_delination = transformed_delineation
            else:
                aggregate_delination = np.add(aggregate_delination, transformed_delineation)
        if aggregate_delination.size == 1:
            print(f"Failed to make composite for patient: {patient}")
            continue

        # Save the aggregate delineation
        majority_vote = (aggregate_delination >= (len(atlas_patients) // 2 + 1)).astype(int)
        # majority_vote = (aggregate_delination >= 1).astype(int)
        advocate_vote = (aggregate_delination >= 2).astype(int)
        voted_image = sitk.GetImageFromArray(advocate_vote)
        voted_image.SetSpacing([0.488281, 0.488281, 1])  # Each pixel is 0.488281 x 0.488281 x 1 mm

        os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists
        output_path = os.path.join(OUTPUT_PATH, f'reg_maj_{patient}.mhd')

        sitk.WriteImage(voted_image, output_path)

    return None

def calc_dice(true_del, est_del):
    # Ensure the arrays are binary (0s and 1s)
    true_del = (true_del > 0).astype(np.uint8)
    est_del = (est_del > 0).astype(np.uint8)

    intersection = np.sum(true_del * est_del)
    size1 = np.sum(true_del)
    size2 = np.sum(est_del)

    if size1 + size2 == 0:
        return 1.0  # If both are empty, define DICE as 1.0 (perfect match)

    return 2.0 * intersection / (size1 + size2)


def find_all_to_all(atlas_patients, register_patients, OUTPUT_PATH, DATA_PATH, TRANSFORMIX_PATH, verbose=False,
                    plot_matrix=False):
    dice_scores = np.zeros((len(atlas_patients), len(register_patients)))

    for j, patient in enumerate(register_patients):
        print(f"Processing Patient: {patient}")

        for i, atlas in enumerate(atlas_patients):
            OUTPUT_DIR = os.path.join(OUTPUT_PATH, f'reg_{patient}_{atlas}')
            transform_path = os.path.join(OUTPUT_DIR, 'TransformParameters.0.txt')

            if not os.path.exists(transform_path):
                dice_scores[i, j] = np.nan  # Indicate missing transformation
                continue

            tr = elastix.TransformixInterface(parameters=transform_path, transformix_path=TRANSFORMIX_PATH)
            moving_delineation_path = os.path.join(DATA_PATH, atlas, 'prostaat.mhd')
            transformed_delineation_path = tr.transform_image(moving_delineation_path, output_dir=OUTPUT_DIR,
                                                              verbose=verbose)

            transformed_delineation = sitk.GetArrayFromImage(sitk.ReadImage(transformed_delineation_path))
            true_delineation = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(DATA_PATH, patient, 'prostaat.mhd')))

            dice_scores[i, j] = calc_dice(transformed_delineation, true_delineation)

    if plot_matrix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(dice_scores, annot=True, fmt=".2f", xticklabels=register_patients, yticklabels=atlas_patients,
                    cmap="viridis")
        plt.xlabel("Registered Patients")
        plt.ylabel("Atlas Patients")
        plt.title("DICE Score Confusion Matrix")
        plt.show()

    return dice_scores