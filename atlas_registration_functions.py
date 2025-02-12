from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk

def register_transform(atlas, patient, DATA_PATH, OUTPUT_PATH, ELASTIX_PATH, TRANSFORMIX_PATH):

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
        parameters=[os.path.join(DATA_PATH, 'parameters_samplespace_MR.txt')],
        output_dir=OUTPUT_DIR,
        verbose=False)

    # Make a new transformix object
    transform_path = os.path.join(OUTPUT_DIR, 'TransformParameters.0.txt')
    tr = elastix.TransformixInterface(parameters=transform_path,
                                      transformix_path=TRANSFORMIX_PATH)

    # Transform the atlas's delineation with the found transformation parameters
    moving_delineation_path = os.path.join(DATA_PATH, atlas, 'prostaat.mhd')
    transformed_delineation_path = tr.transform_image(moving_delineation_path, output_dir=OUTPUT_DIR, verbose=False)

    return transformed_delineation_path
