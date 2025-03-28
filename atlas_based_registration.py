from __future__ import print_function, absolute_import

import os
import warnings
from tqdm import tqdm
from atlas_registration_functions import registrate_atlas_patient, combine_atlas_registrations, find_all_to_all

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths, Make sure the elastix folder etc. is included in your current working directory
paths = dict(line.strip().split("=", 1) for line in open("paths.txt"))
ELASTIX_PATH, TRANSFORMIX_PATH, DATA_PATH, OUTPUT_DIR, PARAM_PATH = (
    paths.get(k) for k in ["elastix_path", "transformix_path", "data_path", "output_path", "paramaters_path"]
)

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')


def register_all_patients(atlas_patients, register_patients, DATA_PATH, OUTPUT_DIR, ELASTIX_PATH, verbose=False):
    # Outer loop: iterate over patients with a progress bar
    for patient in tqdm(register_patients, desc="Processing Patients", unit="patient"):
        # Inner loop: register each patient to all atlas patients with a progress bar
        for atlas in tqdm(atlas_patients, desc=f"Registering {patient}", unit="atlas", leave=False):
            try:
                _ = registrate_atlas_patient(atlas, patient, DATA_PATH, OUTPUT_DIR, ELASTIX_PATH,
                                      verbose)
            except:
                print(f"Failed to register atlas {atlas} to patient {patient}!")
                flag = True
                break

        tqdm.write(f"Registered patient: {patient}.")

    return None


if __name__ == "__main__":
    # Get patient names and select atlas patients
    patient_list = [patient for patient in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, patient))]
    # atlas_patients = patient_list[:5]
    # atlas_patients = ["p135", "p107", "p115"] # low
    # atlas_patients = ["p125", "p120", "p133"] # middle
    # atlas_patients = ["p109", "p108", "p129"] # high
    atlas_patients = ["p135", "p120", "p129"] # combined
    test_patients = ["p137", "p141", "p143", "p144", "p147"] # test set images

    register_patients = [patient for patient in patient_list if patient not in atlas_patients]
    print(atlas_patients)
    print(register_patients)

    # patient_list = [patient for patient in patient_list if patient not in register_patients]

    # Register all the patients
    register_all_patients(atlas_patients, register_patients, DATA_PATH, OUTPUT_DIR, ELASTIX_PATH, verbose=True)

    # Register all to all:
    # register_all_patients(patient_list, patient_list, DATA_PATH, OUTPUT_DIR, ELASTIX_PATH, verbose=True)

    # Combine the registrations
    combine_atlas_registrations(atlas_patients, register_patients, OUTPUT_DIR, DATA_PATH, TRANSFORMIX_PATH)

    #All to all DICE scores:
    # dice_scores, hausdorff, accuracy, precision = find_all_to_all(patient_list, patient_list, OUTPUT_DIR, DATA_PATH, TRANSFORMIX_PATH, verbose=False,
    #                 plot_matrix=True)
    # print("DICE:\n")
    # print(repr(dice_scores))
    #
    # print("hausdorff:\n")
    # print(repr(hausdorff))
    #
    # print("accuracy:\n")
    # print(repr(accuracy))
    #
    # print("precision:\n")
    # print(repr(precision))
