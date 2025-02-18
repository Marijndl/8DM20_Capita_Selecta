import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


class PointSampler:
    def __init__(self, n_points, bounds):
        self.n_points = n_points
        self.bounds = bounds

    def sample_unique_3d_gaussian(self, mean=(0, 0, 0), cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], max_attempts=10000):
        unique_points = set()
        attempts = 0

        while len(unique_points) < self.n_points and attempts < max_attempts:
            sample = np.random.multivariate_normal(mean, cov)
            x, y, z = sample

            if (self.bounds[0][0] <= x <= self.bounds[0][1] and
                    self.bounds[1][0] <= y <= self.bounds[1][1] and
                    self.bounds[2][0] <= z <= self.bounds[2][1]):
                unique_points.add((round(x, 3), round(y, 3), round(z, 3)))

            attempts += 1

        if len(unique_points) < self.n_points:
            raise ValueError("Could not generate enough unique points within the bounds.")

        return [list(point) for point in unique_points]

    def sample_unique_3d_uniform(self):
        points = set()
        while len(points) < self.n_points:
            x = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
            z = np.random.uniform(self.bounds[2][0], self.bounds[2][1])
            points.add((round(x, 3), round(y, 3), round(z, 3)))

        return [list(point) for point in points]

    def sample_unique_3d_beta(self, alpha=2, beta=5):
        points = set()
        while len(points) < self.n_points:
            x = np.random.beta(alpha, beta) * (self.bounds[0][1] - self.bounds[0][0]) + self.bounds[0][0]
            y = np.random.beta(alpha, beta) * (self.bounds[1][1] - self.bounds[1][0]) + self.bounds[1][0]
            z = np.random.beta(alpha, beta) * (self.bounds[2][1] - self.bounds[2][0]) + self.bounds[2][0]
            points.add((round(x, 3), round(y, 3), round(z, 3)))

        return [list(point) for point in points]


if __name__ == "__main__":
    img = sitk.ReadImage(r'D:\capita_selecta\DevelopmentData\DevelopmentData\p102\mr_bffe.mhd')
    x_size, y_size, z_size = img.GetSpacing()[0] * img.GetSize()[0] / 2, img.GetSpacing()[1] * img.GetSize()[1] / 2, \
                             img.GetSpacing()[2] * img.GetSize()[2] / 2
    print(x_size, y_size, z_size)

    n_points = 3000
    mean = (0, 0, 0)
    cov = [[2*x_size, 0, 0], [0, 2*y_size, 0], [0, 0, 2*z_size]]
    cov_large = [[5*x_size, 0, 0], [0, 5*y_size, 0], [0, 0, 5*z_size]]
    bounds = ((-x_size, x_size), (-y_size, y_size), (-z_size, z_size))
    beta_size_1 = 0.5
    beta_size_2 = 0.3


    sampler = PointSampler(n_points=n_points, bounds=bounds)
    points_uniform = sampler.sample_unique_3d_uniform()
    points_gaussian = sampler.sample_unique_3d_gaussian(mean, cov)
    points_gaussian_large_cov = sampler.sample_unique_3d_gaussian(mean, cov_large)
    points_beta = sampler.sample_unique_3d_beta(alpha=beta_size_1, beta=beta_size_1)
    points_beta_small_alpha_beta = sampler.sample_unique_3d_beta(alpha=beta_size_2, beta=beta_size_2)

    print(points_uniform[:5])
    print(points_gaussian[:5])
    print(points_gaussian_large_cov[:5])
    print(points_beta[:5])
    print(points_beta_small_alpha_beta[:5])

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    points_uniform_array = np.array(points_uniform)
    points_gaussian_array = np.array(points_gaussian)
    points_gaussian_large_cov_array = np.array(points_gaussian_large_cov)
    points_beta_array = np.array(points_beta)
    points_beta_small_alpha_beta_array = np.array(points_beta_small_alpha_beta)

    axes[0].scatter(points_uniform_array[:, 0], points_uniform_array[:, 1], s=0.2)
    axes[0].set_title("Sampled from Uniform")
    axes[1].scatter(points_gaussian_array[:, 0], points_gaussian_array[:, 1], s=0.2)
    axes[1].set_title(f"Sampled from Gaussian. Mean: {mean[0]} covariance: {cov[0][0], cov[1][1], cov[2][2]}")
    axes[2].scatter(points_gaussian_large_cov_array[:, 0], points_gaussian_large_cov_array[:, 1], s=0.2)
    axes[2].set_title(f"Sampled from Gaussian. Mean: {mean[0]} covariance: {cov_large[0][0], cov_large[1][1], cov_large[2][2]}")
    axes[3].scatter(points_beta_array[:, 0], points_beta_array[:, 1], s=0.2)
    axes[3].set_title("Sampled from Beta")
    axes[4].scatter(points_beta_small_alpha_beta_array[:, 0], points_beta_small_alpha_beta_array[:, 1], s=0.2)
    axes[4].set_title("Sampled from Beta (Small Alpha & Beta)")

    plt.tight_layout()
    plt.show()

