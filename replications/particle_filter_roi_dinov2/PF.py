import numpy as np
import torch
import matplotlib.pyplot as plt
from meta import proj_3301, read_exif_info
from img_preprocess import crop_and_rotate_roi, crop_black_edges, fix_yaw_if_nadir_roi
from feature_extr import DinoExtractor
from matching import DINOEmbeddingMeasurement
from rasterio.transform import xy, rowcol
from torchfilter.base import DynamicsModel

def uniform_points_in_grid(n_points, transform, grid_size=1024, crop_px=0.0):
    """
    Generate a structured uniform grid of points within a 1024x1024 area
    and convert them to world coordinates.
    """
    if n_points <= 0:
        return np.empty((0, 2), dtype=float)

    # 1. Determine grid dimensions (how many points per side)
    # We use sqrt so that side * side is approximately n_points
    side = int(np.sqrt(n_points))
    side = max(side, 1)
    
    # 2. Compute valid grid bounds
    min_val = crop_px
    max_val = grid_size - crop_px

    # 3. Create evenly spaced coordinates for rows and columns
    rows = np.linspace(min_val, max_val, side)
    cols = np.linspace(min_val, max_val, side)

    # 4. Create the 2D meshgrid
    # row_grid and col_grid will both be (side, side) arrays
    col_grid, row_grid = np.meshgrid(cols, rows)

    # 5. Flatten and convert to world coordinates
    # .ravel() turns the grids into 1D lists of length (side*side)
    points = np.array([
        xy(transform, r, c) 
        for r, c in zip(row_grid.ravel(), col_grid.ravel())
    ])

    return points


def random_points_in_grid(n_points, transform, grid_size=1024, crop_px=0.0):
    """
    Generate random points within a 1024x1024 grid (optionally cropped near edges),
    and convert them to world coordinates using xy().

    Args:
        n_points   : number of random points to generate
        transform  : affine/raster transform used by xy()
        grid_size  : grid size (default 1024)
        crop_px    : width of margin near edges to exclude (in world units)

    Returns:
        (N, 2) numpy array of [x, y] world coordinates
    """

    # Compute valid grid bounds (exclude cropped margins)
    min_row = crop_px
    max_row = grid_size - crop_px
    min_col = crop_px
    max_col = grid_size - crop_px

    # Randomly sample grid coordinates
    rows = np.random.uniform(min_row, max_row, n_points)
    cols = np.random.uniform(min_col, max_col, n_points)

    # Convert grid coordinates to world coordinates
    points = np.array([xy(transform, r, c) for r, c in zip(rows, cols)])

    return points


def randomize_roi_center(xLe, yLe, max_offset_m=20):
    """
    Randomize ROI center around GPS coords.
    
    Args:
        xLe, yLe : float
            True GPS-projected coordinates.
        max_offset_m : float
            Max offset in meters (uniform in both x and y).
    """
    dx = np.random.uniform(-max_offset_m, max_offset_m)
    dy = np.random.uniform(-max_offset_m, max_offset_m)
    return xLe + dx, yLe + dy


class GPSStepDynamics(DynamicsModel):
    def __init__(self, noise_std=1.0):
        super().__init__(state_dim=2)
        self.noise_std = noise_std

    def forward(self, states=None, controls=None, initial_states=None, **kwargs):
        # Use initial_states if states is None
        if states is None:
            if initial_states is None:
                raise ValueError("No state provided for dynamics update")
            states = initial_states

        x, y = states[..., 0], states[..., 1]
        dist, head = controls[..., 0], controls[..., 1]
        dx = dist * torch.cos(head)
        dy = dist * torch.sin(head)
        new_x = x + dx
        new_y = y + dy

        new_state = torch.stack([new_x, new_y], dim=-1)

        # Scale tril for each particle
        scale_tril = torch.eye(self.state_dim, device=new_state.device).unsqueeze(0)
        scale_tril = scale_tril.repeat(new_state.shape[0], 1, 1) * self.noise_std


        # Add Gaussian noise to the state mean for the particle filter
        new_state = new_state + torch.randn_like(new_state) * self.noise_std

        return new_state, scale_tril



def replace_out_of_bounds_particles_grid(particles, transform, crop_px=0, grid_size=1024):
    """
    Replace particles that are outside a 1024x1024 grid or within a band near its edges.

    Args:
        particles : (N, 2) numpy array of [x, y] world coordinates
        transform : raster/affine transform (used by xy() and rowcol())
        crop_px    : width of band near edges to exclude 
        grid_size  : size of grid (default 1024)
    Returns:
        Updated particles array, all within inner grid bounds.
    """

    # Compute inner grid bounds
    min_row = crop_px
    min_col = crop_px
    max_row = grid_size - crop_px
    max_col = grid_size - crop_px

    # Convert particles from world to grid indices
    rows, cols = np.array([rowcol(transform, x, y) for x, y in particles]).T

    # Identify out-of-bounds particles
    mask = (rows < min_row) | (rows >= max_row) | (cols < min_col) | (cols >= max_col)
    num_replace = np.sum(mask)

    if num_replace > 0:
        # Sample replacement pixels within inner region
        new_rows = np.random.uniform(min_row, max_row, num_replace)
        new_cols = np.random.uniform(min_col, max_col, num_replace)

        # Convert back to world coordinates
        new_points = np.array([xy(transform, r, c) for r, c in zip(new_rows, new_cols)])

        # Replace out-of-bounds particles
        particles[mask] = new_points

    return particles



def update_particles(
    particles,
    N_particles,
    roi_resized_transform,
    xLe,
    yLe,
    previous_drone_pos,
    dynamics_model,
):
    """Initialize particles on the first frame or move them using GPS displacement."""

    if particles is None:
        # First image: initialize randomly
        points = random_points_in_grid(
            N_particles, roi_resized_transform, crop_px=128
        )
        xs, ys = points[:, 0], points[:, 1]
        particles = np.stack([xs, ys], axis=1)
    else:
        # Compute GPS-based displacement from previous frame
        dx = xLe - previous_drone_pos[0]
        dy = yLe - previous_drone_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        heading = np.arctan2(dy, dx)

        # Propagate particles using GPSStepDynamics
        particles_torch = torch.tensor(
            particles, dtype=torch.float32
        ).unsqueeze(0)

        controls = torch.tensor(
            [[dist, heading]] * N_particles,
            dtype=torch.float32,
        ).unsqueeze(0)

        particles_torch, _ = dynamics_model.forward(
            states=particles_torch,
            controls=controls,
        )
        particles = particles_torch.squeeze(0).cpu().numpy()

        # Replace particles outside the new bounding box
        particles = replace_out_of_bounds_particles_grid(
            particles, roi_resized_transform, crop_px=128
        )

    # Return updated particles and remember this frame's GPS position
    previous_drone_pos = (xLe, yLe)

    return particles, previous_drone_pos



def estimate_position(particles, weights):
    """Estimate the drone's position as the weighted average of particles."""

    weights = np.asarray(weights).reshape(-1)
    x_est = np.sum(particles[:, 0] * weights)
    y_est = np.sum(particles[:, 1] * weights)

    return x_est, y_est


def resample_particles(particles, weights):
    """Resample particles according to their visual-matching weights."""

    indices = np.random.choice(
        len(particles),
        size=len(particles),
        p=weights,
    )

    return particles[indices]


def update_particle_population(
    particles,
    weights,
    N_particles,
    N_scout_particles,
    roi_resized_transform,
):
    """Resample tracker particles and add scout particles for the next frame."""

    scout_count = max(N_scout_particles, 0)
    tracker_count = max(N_particles - scout_count, 0)

    # Resample tracker particles according to their weights
    if tracker_count > 0:
        indices = np.random.choice(
            N_particles,
            size=tracker_count,
            p=weights,
        )
        trackers = particles[indices]
    else:
        trackers = np.empty((0, 2), dtype=particles.dtype)

    # Generate scout particles across the ROI
    if scout_count > 0:
        scouts = uniform_points_in_grid(
            scout_count,
            roi_resized_transform,
            grid_size=1024,
            crop_px=128,
        )
    else:
        scouts = np.empty((0, 2), dtype=particles.dtype)

    # Combine the two populations
    if trackers.size == 0 and scouts.size == 0:
        updated_particles = np.empty((0, 2), dtype=particles.dtype)
    elif trackers.size == 0:
        updated_particles = scouts
    elif scouts.size == 0:
        updated_particles = trackers
    else:
        updated_particles = np.vstack([trackers, scouts])

    # Preserve the existing behavior when the population is too small
    if len(updated_particles) < N_particles:
        gap = N_particles - len(updated_particles)

        if len(trackers) > 0:
            extra_points = trackers[:gap]
            updated_particles = np.vstack([updated_particles, extra_points])
        elif len(updated_particles) > 0:
            extra_points = updated_particles[:gap]
            updated_particles = np.vstack([updated_particles, extra_points])

    return updated_particles

