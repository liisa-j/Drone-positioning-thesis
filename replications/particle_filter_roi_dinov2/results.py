"""Functions for saving localization results and visualisations."""

import os
import csv
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from rasterio.transform import rowcol
import numpy as np


def get_unique_output_dir(base_folder, flight, model_name, method):
    """Create a new output directory without overwriting previous results."""
    counter = 0

    while True:
        folder_name = f"{flight}_{method}_{model_name}_outputs"

        if counter > 0:
            folder_name += f"_{counter}"

        output_dir = os.path.join(base_folder, folder_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            return output_dir

        counter += 1


class CsvResultsWriter:
    """Write particle-filter experiment results to a tab-separated file."""

    def __init__(
        self,
        output_dir,
        flight,
        method,
        model_name,
        n_particles,
        scout_ratio,
        dynamic_noise_std,
        measurement_model_temp,
    ):
        self.csv_path = os.path.join(
            output_dir,
            f"{flight}_{method}_{model_name}_results.csv",
        )

        self.csv_file = open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.writer(self.csv_file, delimiter="\t")

        # Experiment configuration
        self.writer.writerow(["N_particles", n_particles])
        self.writer.writerow(["scout_ratio", scout_ratio])
        self.writer.writerow(["dynamic_noise_std", dynamic_noise_std])
        self.writer.writerow(["measurment_model_temp", measurement_model_temp])
        self.writer.writerow([])
        self.writer.writerow([
            "image_name",
            "offset_m",
            "edge_length_m",
            "inference_time_s",
        ])

    def write_row(self, image_name, offset_m, edge_length_m, inference_time_s):
        """Write the results for one image."""
        self.writer.writerow([
            image_name,
            offset_m,
            edge_length_m,
            inference_time_s,
        ])

    def close(self):
        """Close the CSV file."""
        self.csv_file.close()



def save_particle_plot(
    roi_img,
    particles,
    roi_transform,
    est_x,
    est_y,
    drone_x,
    drone_y,
    drone_img,
    image_name,
    output_dir,
):
    """Save the particle-filter visualisation for one image."""

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(roi_img)

    pxs, pys = [], []
    for x, y in particles:
        py, px = rowcol(roi_transform, x, y)
        pxs.append(px)
        pys.append(py)

    axs[0].scatter(
        pxs, pys, c="red", s=10, alpha=0.4, label="Particles"
    )

    est_py, est_px = rowcol(roi_transform, est_x, est_y)
    axs[0].scatter(
        [est_px], [est_py],
        c="lime", s=60, label="Estimate", marker="x"
    )

    drone_py, drone_px = rowcol(roi_transform, drone_x, drone_y)
    axs[0].scatter(
        [drone_px], [drone_py],
        c="blue", s=60, label="Drone (true)", marker="o"
    )

    axs[0].set_title(f"Ortho + particles for {image_name}")
    axs[0].legend()

    axs[1].imshow(drone_img)
    axs[1].set_title("Drone image")
    axs[1].axis("off")

    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"{image_name}_particles.png")
    plt.savefig(fig_path)
    plt.close(fig)



def save_trajectory_and_error_plots(
    est_path,
    gps_path,
    offsets,
    ortho_dataset,
    ortho_img_rgb,
    output_dir,
    flight,
    method,
    model_name,
):
    """Save the cropped trajectory plot and localization error plot."""

    from rasterio.transform import rowcol

    PATH_PADDING_M = 50.0

    # Convert paths to numpy arrays
    est_path_np = np.array(est_path)
    gps_path_np = np.array(gps_path)

    all_x = np.concatenate([est_path_np[:, 0], gps_path_np[:, 0]])
    all_y = np.concatenate([est_path_np[:, 1], gps_path_np[:, 1]])

    # Bounding box in world coordinates
    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()

    # Make the crop square
    width = max_x - min_x
    height = max_y - min_y
    side = max(width, height) + 2 * PATH_PADDING_M

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    half = side / 2
    crop_min_x = cx - half
    crop_max_x = cx + half
    crop_min_y = cy - half
    crop_max_y = cy + half

    # Convert world coordinates to pixel indices
    row_min, col_min = rowcol(
        ortho_dataset.transform, crop_min_x, crop_max_y
    )
    row_max, col_max = rowcol(
        ortho_dataset.transform, crop_max_x, crop_min_y
    )

    # Clamp to image bounds
    H, W, _ = ortho_img_rgb.shape
    row_min = np.clip(row_min, 0, H - 1)
    row_max = np.clip(row_max, 0, H - 1)
    col_min = np.clip(col_min, 0, W - 1)
    col_max = np.clip(col_max, 0, W - 1)

    cropped_ortho = ortho_img_rgb[row_min:row_max, col_min:col_max]

    # Draw trajectory plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cropped_ortho)

    def plot_path(path, color, label, linestyle="-"):
        pxs, pys = [], []

        for x, y in path:
            py, px = rowcol(ortho_dataset.transform, x, y)
            pxs.append(px - col_min)
            pys.append(py - row_min)

        ax.plot(
            pxs, pys,
            color=color,
            linewidth=2,
            linestyle=linestyle,
            label=label,
        )
        return pxs, pys

    gps_pxs, gps_pys = plot_path(gps_path, "blue", "GPS path")
    est_pxs, est_pys = plot_path(
        est_path, "lime", "Estimated path", linestyle="--"
    )

    # GPS start and end
    ax.scatter(
        gps_pxs[0], gps_pys[0],
        c="cyan", s=80, marker="o",
        label="GPS start", zorder=5,
    )
    ax.scatter(
        gps_pxs[-1], gps_pys[-1],
        c="blue", s=80, marker="X",
        label="GPS end", zorder=5,
    )

    # Estimated start and end
    ax.scatter(
        est_pxs[0], est_pys[0],
        c="yellow", s=80, marker="o",
        label="Est start", zorder=5,
    )
    ax.scatter(
        est_pxs[-1], est_pys[-1],
        c="lime", s=80, marker="X",
        label="Est end", zorder=5,
    )

    ax.set_title("Trajectory (cropped ortho)")
    ax.legend()
    ax.axis("off")

    cropped_path_img = os.path.join(
        output_dir,
        f"{flight}_{method}_{model_name}_trajectory_cropped.png",
    )

    plt.savefig(cropped_path_img, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Cropped trajectory saved to {cropped_path_img}")

    # Draw localization error plot
    fig, ax = plt.subplots(figsize=(10, 4))

    frames = np.arange(len(offsets))
    ax.plot(frames, offsets, linewidth=2)

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Localization error [m]")
    ax.set_title("Particle filter localization error over time")
    ax.grid(True, alpha=0.3)

    error_plot_path = os.path.join(
        output_dir,
        f"{flight}_{method}_{model_name}_error_plot.png",
    )

    plt.tight_layout()
    plt.savefig(error_plot_path, dpi=200)
    plt.close(fig)

    print(f"Error plot saved to {error_plot_path}")


def calculate_offset(est_x, est_y, gps_x, gps_y):
    """Calculate the distance in meters between estimated and GPS positions."""
    return np.sqrt((est_x - gps_x) ** 2 + (est_y - gps_y) ** 2)