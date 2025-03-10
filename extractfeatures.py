import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

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


    def create_percentile_groups(self,lowerlimit=25,upperlimit=75):
        """
        Select the lowest n-th percentile, n-th to m-th percentile, and m-th to 100th percentile of each dictionary.

        Parameters:
        - n: The lower percentile (default 25th percentile)
        - m: The upper percentile (default 75th percentile)

        Returns:
        - A dictionary containing three groups for each percentile range.
        """
        # Ensure that dictionaries are sorted by their values
        sorted_patient_ids = list(self.patient_svrs.keys())  # Use sorted patient IDs
        svr_values = np.array([self.patient_svrs[patient_id] for patient_id in sorted_patient_ids])
        volume_values = np.array([self.patient_volumes[patient_id] for patient_id in sorted_patient_ids])
        heterogeneity_values = np.array([self.patient_heterogeneity[patient_id] for patient_id in sorted_patient_ids])

        # Calculate the percentiles
        svr_nth_percentile = np.percentile(svr_values, lowerlimit)
        svr_mth_percentile = np.percentile(svr_values, upperlimit)

        volume_nth_percentile = np.percentile(volume_values, lowerlimit)
        volume_mth_percentile = np.percentile(volume_values, upperlimit)

        heterogeneity_nth_percentile = np.percentile(heterogeneity_values, lowerlimit)
        heterogeneity_mth_percentile = np.percentile(heterogeneity_values, upperlimit)

        # Divide the data into three percentile ranges
        svr_low = {patient_id: self.patient_svrs[patient_id] for patient_id in sorted_patient_ids if
                   self.patient_svrs[patient_id] <= svr_nth_percentile}
        svr_middle = {patient_id: self.patient_svrs[patient_id] for patient_id in sorted_patient_ids if
                      svr_nth_percentile < self.patient_svrs[patient_id] <= svr_mth_percentile}
        svr_high = {patient_id: self.patient_svrs[patient_id] for patient_id in sorted_patient_ids if
                    self.patient_svrs[patient_id] > svr_mth_percentile}

        volume_low = {patient_id: self.patient_volumes[patient_id] for patient_id in sorted_patient_ids if
                      self.patient_volumes[patient_id] <= volume_nth_percentile}
        volume_middle = {patient_id: self.patient_volumes[patient_id] for patient_id in sorted_patient_ids if
                         volume_nth_percentile < self.patient_volumes[patient_id] <= volume_mth_percentile}
        volume_high = {patient_id: self.patient_volumes[patient_id] for patient_id in sorted_patient_ids if
                       self.patient_volumes[patient_id] > volume_mth_percentile}

        heterogeneity_low = {patient_id: self.patient_heterogeneity[patient_id] for patient_id in sorted_patient_ids if
                             self.patient_heterogeneity[patient_id] <= heterogeneity_nth_percentile}
        heterogeneity_middle = {patient_id: self.patient_heterogeneity[patient_id] for patient_id in sorted_patient_ids
                                if heterogeneity_nth_percentile < self.patient_heterogeneity[
                                    patient_id] <= heterogeneity_mth_percentile}
        heterogeneity_high = {patient_id: self.patient_heterogeneity[patient_id] for patient_id in sorted_patient_ids if
                              self.patient_heterogeneity[patient_id] > heterogeneity_mth_percentile}

        # Create a dictionary containing the groups
        percentile_groups = {
            'SVR': {'low': svr_low, 'middle': svr_middle, 'high': svr_high},
            'Volume': {'low': volume_low, 'middle': volume_middle, 'high': volume_high},
            'Heterogeneity': {'low': heterogeneity_low, 'middle': heterogeneity_middle, 'high': heterogeneity_high}
        }

        return percentile_groups