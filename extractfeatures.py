import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
class extractfeatures:
    def __init__(self,DATA_PATH):
        self.patient_ids = [patient_id for patient_id in os.listdir(DATA_PATH) if patient_id.startswith("p")]
        self.patient_imgs = []
        self.patient_delins = []

        for patient_id in self.patient_ids:
            patient_folder = os.path.join(DATA_PATH, patient_id)

            # Paths for the specific files
            mr_bffe_path = os.path.join(patient_folder, "mr_bffe.mhd")
            prostaat_path = os.path.join(patient_folder, "prostaat.mhd")

            if os.path.exists(mr_bffe_path):
                sitk_image = sitk.ReadImage(mr_bffe_path)
                self.patient_imgs.append(sitk.GetArrayFromImage(sitk_image))  # Convert to NumPy

            if os.path.exists(prostaat_path):
                sitk_mask = sitk.ReadImage(prostaat_path)
                self.patient_delins.append(sitk.GetArrayFromImage(sitk_mask))  # Convert to NumPy

        self.voxelsize = sitk_image.GetSpacing()


    def calculate_svr(self):
        """Bereken Surface-to-Volume Ratio (SVR)."""

        patient_svrs = {}

        for idx, mask in enumerate(self.patient_delins):
            verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0)
            S = measure.mesh_surface_area(verts, faces)  # Oppervlakte
            V = np.sum(mask)  # Volume = aantal voxels
            patient_svrs[self.patient_ids[idx]] = S/V

        self.patient_svrs = dict(sorted(patient_svrs.items(), key=lambda item: item[1]))

        return self.patient_svrs

    def calculate_volume(self):

        patient_volumes = {}
        voxel_volume = np.prod(self.voxelsize)  #Computes volume of 1 voxel in mm^3

        for idx, mask in enumerate(self.patient_delins):
            voxel_count = np.sum(mask)
            volume = voxel_volume * voxel_count
            patient_volumes[self.patient_ids[idx]] = volume

        self.patient_volumes = dict(sorted(patient_volumes.items(), key=lambda item: item[1]))

        return self.patient_volumes

    def calculate_heterogeneity(self):

        patient_heterogeneity = {}

        for idx, mask in enumerate(self.patient_delins):
            tumor_voxels = self.patient_imgs[idx][mask == 1]
            patient_heterogeneity[self.patient_ids[idx]] = np.std(tumor_voxels)

        self.patient_heterogeneity = dict(sorted(patient_heterogeneity.items(), key=lambda item: item[1]))

        return self.patient_heterogeneity

    def calculate_product_values(self):

        patient_product_values = {}

        for patientid in self.patient_ids:
            patient_product_values[patientid] = self.patient_heterogeneity[patientid] * self.patient_volumes[patientid]

        self.patient_product_values = dict(sorted(patient_product_values.items(), key=lambda item: item[1]))

        # Generates a duplicate dict with normalized values
        values = np.array(list(patient_product_values.values())).reshape(-1, 1)
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(values).flatten()
        self.patient_product_values_normalized = dict(
            sorted(zip(patient_product_values.keys(), normalized_values), key=lambda item: item[1])
        )

        return self.patient_product_values, self.patient_product_values_normalized

    def generate_2d_plots(self):
        """Generate scatter plots for SVR, volume, and heterogeneity as subplots."""
        # Sort keys of all dictionaries to ensure alignment
        sorted_patient_ids = list(self.patient_svrs.keys())  # Use sorted patient IDs
        svr_values = [self.patient_svrs[patient_id] for patient_id in sorted_patient_ids]
        volume_values = [self.patient_volumes[patient_id] for patient_id in sorted_patient_ids]
        heterogeneity_values = [self.patient_heterogeneity[patient_id] for patient_id in sorted_patient_ids]

        # Create a figure with 1 row and 3 columns of subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Scatter Plot 1: SVR vs Volume
        axes[0].scatter(svr_values, volume_values, color='b')
        axes[0].set_xlabel('SVR')
        axes[0].set_ylabel('Volume (mm^3)')
        axes[0].set_title('SVR vs Volume')
        axes[0].grid(True)

        # Scatter Plot 2: SVR vs Heterogeneity
        axes[1].scatter(svr_values, heterogeneity_values, color='r')
        axes[1].set_xlabel('SVR')
        axes[1].set_ylabel('Heterogeneity')
        axes[1].set_title('SVR vs Heterogeneity')
        axes[1].grid(True)

        # Scatter Plot 3: Volume vs Heterogeneity
        axes[2].scatter(volume_values, heterogeneity_values, color='g')
        axes[2].set_xlabel('Volume (mm^3)')
        axes[2].set_ylabel('Heterogeneity')
        axes[2].set_title('Volume vs Heterogeneity')
        axes[2].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plots
        plt.show()

    def generate_normalized_plots(self):
        # Sort keys of all dictionaries to ensure alignment
        sorted_patient_ids = list(self.patient_svrs.keys())  # Use sorted patient IDs
        svr_values = [self.patient_svrs[patient_id] for patient_id in sorted_patient_ids]
        volume_values = [self.patient_volumes[patient_id] for patient_id in sorted_patient_ids]
        heterogeneity_values = [self.patient_heterogeneity[patient_id] for patient_id in sorted_patient_ids]

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Reshape the lists to fit scaler requirements and scale them
        svr_values = scaler.fit_transform([[v] for v in svr_values]).flatten()
        volume_values = scaler.fit_transform([[v] for v in volume_values]).flatten()
        heterogeneity_values = scaler.fit_transform([[v] for v in heterogeneity_values]).flatten()

        # Compute the product of volume and heterogeneity
        product_values = volume_values * heterogeneity_values

        # Create a **new** figure for the plot (this ensures no previous figure is reused)
        plt.figure(figsize=(20, 5))  # Creates a new figure instance

        # Scatter Plot 1: SVR vs Volume
        plt.subplot(1, 4, 1)
        plt.scatter(svr_values, volume_values, color='b')
        plt.xlabel('SVR (normalized)')
        plt.ylabel('Volume (normalized)')
        plt.title('SVR vs Volume')
        plt.grid(True)

        # Scatter Plot 2: SVR vs Heterogeneity
        plt.subplot(1, 4, 2)
        plt.scatter(svr_values, heterogeneity_values, color='r')
        plt.xlabel('SVR (normalized)')
        plt.ylabel('Heterogeneity (normalized)')
        plt.title('SVR vs Heterogeneity')
        plt.grid(True)

        # Scatter Plot 3: Volume vs Heterogeneity
        plt.subplot(1, 4, 3)
        plt.scatter(volume_values, heterogeneity_values, color='g')
        plt.xlabel('Volume (normalized)')
        plt.ylabel('Heterogeneity (normalized)')
        plt.title('Volume vs Heterogeneity')
        plt.grid(True)

        # Vertical Scatter Plot 4: Product of Volume × Heterogeneity
        plt.subplot(1, 4, 4)
        plt.scatter([0] * len(product_values), product_values, color='purple', alpha=0.7)  # All points at x=0
        plt.xlabel('X (constant)')
        plt.ylabel('Volume × Heterogeneity (normalized)')
        plt.title('Vertical Scatter Plot of Volume × Heterogeneity')
        plt.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plots
        plt.show()

    def generate_vertical_plot(self):
        """Generates the vertical scatter plot of Volume × Heterogeneity with log scale on Y-axis."""

        # Sort keys of all dictionaries to ensure alignment
        sorted_patient_ids = list(self.patient_svrs.keys())  # Use sorted patient IDs
        volume_values = [self.patient_volumes[patient_id] for patient_id in sorted_patient_ids]
        heterogeneity_values = [self.patient_heterogeneity[patient_id] for patient_id in sorted_patient_ids]

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Reshape the lists to fit scaler requirements and scale them
        volume_values = scaler.fit_transform([[v] for v in volume_values]).flatten()
        heterogeneity_values = scaler.fit_transform([[v] for v in heterogeneity_values]).flatten()

        # Create the product of volume and heterogeneity
        product_values = volume_values * heterogeneity_values  # This is the new variable

        # Create a new figure for the vertical scatter plot
        plt.figure(figsize=(10, 5))  # Creates a new figure instance

        # Vertical Scatter Plot: Product of Volume × Heterogeneity
        plt.scatter([0] * len(product_values), product_values, color='purple', alpha=0.7)  # All points at x=0
        plt.xlabel('X (constant)')
        plt.ylabel('Volume × Heterogeneity (normalized)')
        plt.title('Vertical Scatter Plot of Volume × Heterogeneity')

        # Apply log scale to the Y-axis
        plt.yscale('log')

        # Set a minimum limit to avoid log(0) issues
        plt.ylim(bottom=1e-3)  # Adjust this value based on your data

        plt.grid(True)

        # Show the plot
        plt.show()
    def generate_3d_plot(self):
        """Generate a 3D scatter plot for SVR, volume, and heterogeneity."""
        # Sort keys of all dictionaries to ensure alignment
        sorted_patient_ids = list(self.patient_svrs.keys())  # Use sorted patient IDs
        svr_values = [self.patient_svrs[patient_id] for patient_id in sorted_patient_ids]
        volume_values = [self.patient_volumes[patient_id] for patient_id in sorted_patient_ids]
        heterogeneity_values = [self.patient_heterogeneity[patient_id] for patient_id in sorted_patient_ids]

        # Create a 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot: SVR vs Volume vs Heterogeneity
        ax.scatter(svr_values, volume_values, heterogeneity_values, c='r', marker='o')

        # Set labels for the axes
        ax.set_xlabel('SVR')
        ax.set_ylabel('Volume (mm^3)')
        ax.set_zlabel('Heterogeneity')

        # Set title for the plot
        ax.set_title('3D Plot: SVR, Volume, Heterogeneity')

        # Show the plot
        plt.show()

    def create_percentile_groups(self, lowerlimit=25, upperlimit=75):
        """
        Select the lowest n-th percentile, n-th to m-th percentile, and m-th to 100th percentile for each metric.

        Parameters:
        - lowerlimit: The lower percentile (default 25th percentile)
        - upperlimit: The upper percentile (default 75th percentile)

        Returns:
        - A dictionary containing three groups (low, middle, high) for each metric.
        """

        def compute_percentiles(data_dict):
            """Sorts patients based on a metric, computes percentile cutoffs, and groups them."""
            sorted_items = sorted(data_dict.items(), key=lambda x: x[1])  # Sort dictionary by values
            sorted_patient_ids = [pid for pid, _ in sorted_items]
            sorted_values = np.array([value for _, value in sorted_items])

            # Calculate percentile cutoffs
            nth_percentile = np.percentile(sorted_values, lowerlimit)
            mth_percentile = np.percentile(sorted_values, upperlimit)

            # Group patients based on the percentile thresholds
            low = {pid: data_dict[pid] for pid in sorted_patient_ids if data_dict[pid] <= nth_percentile}
            middle = {pid: data_dict[pid] for pid in sorted_patient_ids if
                      nth_percentile < data_dict[pid] <= mth_percentile}
            high = {pid: data_dict[pid] for pid in sorted_patient_ids if data_dict[pid] > mth_percentile}

            return low, middle, high

        # Compute percentile groups separately for each metric
        svr_low, svr_middle, svr_high = compute_percentiles(self.patient_svrs)
        volume_low, volume_middle, volume_high = compute_percentiles(self.patient_volumes)
        heterogeneity_low, heterogeneity_middle, heterogeneity_high = compute_percentiles(self.patient_heterogeneity)
        product_low, product_middle, product_high = compute_percentiles(self.patient_product_values)
        product_norm_low, product_norm_middle, product_norm_high = compute_percentiles(
            self.patient_product_values_normalized)

        # Create a dictionary containing the percentile groups for all metrics
        percentile_groups = {
            'SVR': {'low': svr_low, 'middle': svr_middle, 'high': svr_high},
            'Volume': {'low': volume_low, 'middle': volume_middle, 'high': volume_high},
            'Heterogeneity': {'low': heterogeneity_low, 'middle': heterogeneity_middle, 'high': heterogeneity_high},
            'Product': {'low': product_low, 'middle': product_middle, 'high': product_high},
            'Product_Normalized': {'low': product_norm_low, 'middle': product_norm_middle, 'high': product_norm_high}
        }

        return percentile_groups