from __future__ import print_function, absolute_import

import os
import warnings
from tqdm import tqdm
from atlas_registration_functions import registrate_atlas_patient, combine_atlas_registrations, find_all_to_all

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths
ELASTIX_PATH = r'D:\Elastix\elastix.exe'
TRANSFORMIX_PATH = r'D:\Elastix\transformix.exe'
DATA_PATH = r'D:\capita_selecta\DevelopmentData\DevelopmentData'
OUTPUT_PATH = r'D:\capita_selecta\results_all_to_all'

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')


def register_all_patients(atlas_patients, register_patients, DATA_PATH, OUTPUT_PATH, ELASTIX_PATH, verbose=False):
    # Outer loop: iterate over patients with a progress bar
    flag = False
    for patient in tqdm(register_patients, desc="Processing Patients", unit="patient"):
        # Inner loop: register each patient to all atlas patients with a progress bar
        for atlas in tqdm(atlas_patients, desc=f"Registering {patient}", unit="atlas", leave=False):
            try:
                _ = registrate_atlas_patient(atlas, patient, DATA_PATH, OUTPUT_PATH, ELASTIX_PATH,
                                      verbose)
            except:
                print(f"Failed to register atlas {atlas} to patient {patient}!")
                flag = True
                break

        # if flag: # Break out of outer loop
        #     flag = False
        #     continue
        tqdm.write(f"Registered patient: {patient}.")

    return None


if __name__ == "__main__":
    # Get patient names and select atlas patients
    patient_list = [patient for patient in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, patient))]
    # atlas_patients = patient_list[:5]
    atlas_patients = ["p102", "p108", "p109"]
    register_patients = [patient for patient in patient_list if patient not in atlas_patients]

    # Register all the patients
    # register_all_patients(atlas_patients, register_patients, DATA_PATH, OUTPUT_PATH, ELASTIX_PATH, verbose=True)

    # Register all to all:
    # register_all_patients(patient_list, patient_list, DATA_PATH, OUTPUT_PATH, ELASTIX_PATH, verbose=True)

    # Combine the registrations
    # combine_atlas_registrations(atlas_patients, register_patients, OUTPUT_PATH, DATA_PATH, TRANSFORMIX_PATH)

    #All to all DICE scores:
    find_all_to_all(patient_list, patient_list, OUTPUT_PATH, DATA_PATH, TRANSFORMIX_PATH, verbose=False,
                    plot_matrix=True)
