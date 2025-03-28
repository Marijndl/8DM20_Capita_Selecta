# Repository for Medical Image Registration and Analysis
This repository contains code for performing atlas-based segmentation using registration with Elastix. It includes scripts for both image registration and statistical analysis.

This work was conducted for the course 8DM20: Capita Selecta for Medical Imaging in the Department of Biomedical Engineering at Eindhoven University of Technology.

## Table of Contents
- [Authors](#authors)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Atlas-Based Registration](#atlas-based-registration)
  - [Feature Extraction](#feature-extraction)
- [Files](#files)
- [License](#license)

## Authors

Willem Pladet* (1606492)\
Joris Mentink* (1614568)\
Joost Klis* (1503715)\
Marijn de Lange* (1584944)\
Bruno Rütten* (1579320)\
Noach Schilt* (1584979)

*: Technical University Eindhoven, Eindhoven, The Netherlands

## Directory Structure

```
archive/ # Older or experimental scripts
examples/ # Example scripts for testing
parameter_files/ # Parameter files for Elastix registration
.gitignore # Specifies intentionally untracked files that Git should ignore
atlas_based_registration.py # Main script for atlas-based registration
atlas_registration_functions.py # Functions for atlas registration tasks
extractfeatures.py # Script for extracting features from images
paths.txt # Stores paths to Elastix, data, etc.
```

## Requirements

-   Python 3.x
-   SimpleITK
-   Elastix (version >= 4.8)
-   Transformix (comes with Elastix)
-   matplotlib
-   numpy
-   imageio
-   scikit-image
-   scikit-learn
-   tqdm
-   pyvista (for deformation field visualization)
-   seaborn
-   tensorflow

You can install most of the Python dependencies using pip:

```bash
pip install SimpleITK matplotlib numpy imageio scikit-image scikit-learn tqdm pyvista seaborn tensorflow
```

**Important:**

-   Ensure that Elastix and Transformix are correctly installed and the paths are set correctly in `paths.txt` or within the scripts.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Marijndl/8DM20_Capita_Selecta
    cd 8DM20_Capita_Selecta
    ```

2.  Install the required Python packages (see [Requirements](#requirements)).

3.  Configure the paths to Elastix and Transformix in `paths.txt` or directly in the scripts:

    ```python
    ELASTIX_PATH = r'D:\Elastix\elastix.exe'
    TRANSFORMIX_PATH = r'D:\Elastix\transformix.exe'
    ```

    Alternatively, the `paths.txt` file allows you to set these and other paths:

    ```
    elastix_path=D:\Elastix\elastix.exe
    transformix_path=D:\Elastix\transformix.exe
    data_path=D:\capita_selecta\DevelopmentData\DevelopmentData
    output_path=D:\capita_selecta\results_all_to_all
    paramaters_path=C:\Users\20203226\Documents\GitHub\8DM20_Capita_Selecta\parameter_files
    ```

## Usage

### Atlas-Based Registration

The `atlas_based_registration.py` script performs atlas-based registration using Elastix.

1.  **Data Preparation:** Ensure your data is organized with each patient's data in a separate folder under the `DATA_PATH`.  Each patient folder should contain `mr_bffe.mhd` (the image) and `prostaat.mhd` (the delineation).

2.  **Configuration:**
    *   Set the `DATA_PATH` and `OUTPUT_PATH` variables in `paths.txt`.
    *   Choose a set of `atlas_patients` and `register_patients` from the available patient IDs.  The `atlas_patients` are used as the basis for registration, and the `register_patients` are the patients to be registered to the atlases.

3.  **Execution:** Run the script:

    ```bash
    python atlas_based_registration.py
    ```

    This will:

    *   Register each `register_patient` to each `atlas_patient`.
    *   Combine the registrations to create a majority vote delineation.

### Feature Extraction

The `extractfeatures_implementation.ipynb` notebook extracts features from the images.

1.  **Data Preparation:** The script expects the same data organization as the atlas-based registration.

2.  **Configuration:** Modify the `DATA_PATH` variable in the script.

3.  **Execution:** Run the different cells in the notebook to extract the features, calculate the performance of the performed registrations and then find correlations.

    This will:

    *   Calculate the surface-to-volume ratio (SVR), volume, and heterogeneity for each patient.
    *   Generate scatter plots to visualize the relationships between these features.
    *   Calculate the DICE score, Hausdorff distance, accuracy, and precision of the performed registrations.
    *   Calculate the Spearmans correlation between certain features and several performance metrics.

## Files

-   `examples/Example_script.py`: Example script demonstrating Elastix registration.  Shows how to perform image registration using Elastix, transform images, and extract Jacobian determinants.
-   `examples/plot_deformation.py`: Script for plotting deformation fields using PyVista.
-   `parameter_files/Par0001bspline16.txt`: Elastix parameter file for B-spline registration.
-   `parameter_files/Par0001translation.txt`: Elastix parameter file for translation registration.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
-   `atlas_based_registration.py`: Main script for atlas-based registration.
-   `atlas_registration_functions.py`: Functions for atlas registration tasks (registration, combining results, calculating Dice score).  Includes functions to registrate atlases to patients, combine those registrations through majority voting to form a composite delineation, and calculate DICE scores.
-   `extractfeatures.py`: Script for extracting features (volume, SVR, heterogeneity) from images.
-   `paths.txt`: File for storing paths to Elastix, data, etc.
