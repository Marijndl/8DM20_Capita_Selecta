from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk

def register_transform(patient_1, patient_2, DATA_PATH, ELASTIX_PATH, TRANSFORMIX_PATH):

    OUTPUT_DIR = fr'D:\capita_selecta\results_experiments\reg_{patient_1}_{patient_2}'
    # Make a results directory if non exists
    if os.path.exists(OUTPUT_DIR) is False:
        os.mkdir(OUTPUT_DIR)

    # Define the paths to the two images
    fixed_image_path = os.path.join(DATA_PATH, patient_1, 'mr_bffe.mhd')
    moving_image_path = os.path.join(DATA_PATH, patient_2, 'mr_bffe.mhd')

    # Define a new elastix object 'el' with the correct path to elastix
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

    # Execute the registration. Make sure the paths below are correct, and
    # that the results folder exists from where you are running this script
    el.register(
        fixed_image=fixed_image_path,
        moving_image=moving_image_path,
        parameters=[os.path.join(DATA_PATH, 'parameters_samplespace_MR.txt')],
        output_dir=OUTPUT_DIR,
        verbose=False)

    # Make a new transformix object tr with the CORRECT PATH to transformix
    transform_path = os.path.join(OUTPUT_DIR, 'TransformParameters.0.txt')
    tr = elastix.TransformixInterface(parameters=transform_path,
                                      transformix_path=TRANSFORMIX_PATH)

    # Transform a new image with the transformation parameters
    moving_delineation_path = os.path.join(DATA_PATH, patient_2, 'prostaat.mhd')
    transformed_delineation_path = tr.transform_image(moving_delineation_path, output_dir=OUTPUT_DIR, verbose=False)

    return transformed_delineation_path
