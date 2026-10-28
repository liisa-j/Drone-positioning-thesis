import numpy as np
import torch
import torch.nn.functional as F
from meta import pixel2lest
from torchfilter.base import ParticleFilterMeasurementModel
from rasterio.transform import rowcol


def compute_heatmap_ncc(mapF, droneF):
    cross_term = F.conv2d(mapF, droneF, bias=None, stride=1)
    norm_drone = torch.sqrt(torch.sum(droneF**2))

    map_sq = mapF * mapF
    ones_ker = torch.ones_like(droneF)
    sum_map_sq = F.conv2d(map_sq, ones_ker, bias=None, stride=1)
    norm_map = torch.sqrt(sum_map_sq + 1e-8)

    ncc = cross_term / (norm_drone * norm_map + 1e-8)
    heatmap = ncc.squeeze(0).squeeze(0).cpu().numpy()
    return heatmap


def world_to_heatmap(transform, Wf, Hf, W_feat, H_feat, patch_size, x_world, y_world, roi_size=1024):
    """
    Convert (x_world, y_world) → (hx, hy) on the similarity heatmap.

    bbox        : [xmin, ymin, xmax, ymax] of ROI in world coordinates
    Wf, Hf      : width, height of the feature map
    W_feat, H_feat : width, height of the heatmap
    patch_size  : patch size in feature-map pixels (e.g., 49)
    x_world, y_world : world coordinates to map
    """

    # Convert world → feature-map pixel coordinates
    py, px = rowcol(transform, x_world, y_world)

    px_f = (px / roi_size) * Wf
    py_f = (py / roi_size) * Hf

    # Compute effective stride from dimensions
    stride_x = (Wf - patch_size) / (W_feat - 1)
    stride_y = (Hf - patch_size) / (H_feat - 1)

    # Offset due to patch centers (first valid position is at patch_size/2)
    offset_x = patch_size / 2
    offset_y = patch_size / 2

    # Convert feature coords → heatmap coords
    hx = (px_f - offset_x) / stride_x
    hy = (py_f - offset_y) / stride_y

    # Clip to heatmap bounds
    hx = int(np.clip(round(hx), 0, W_feat - 1))
    hy = int(np.clip(round(hy), 0, H_feat - 1))

    return hx, hy


class DINOEmbeddingMeasurement(ParticleFilterMeasurementModel):
    def __init__(self, mapF, transform, drone_vec,
                 crop_px=256, temperature=0.07, oob_penalty=-6.0):
        super().__init__(state_dim=2)
        self.mapF = mapF
        self.transform = transform
        self.drone_vec = drone_vec
        self.crop_px = crop_px
        self.temperature = temperature
        self.oob_penalty = oob_penalty

    @torch.no_grad()
    def forward(self, particle_states):
        """
        particle_states: (N, M, 2) -> candidate positions in world coords
        observations: (N, D) -> drone embedding
        """
        N, M, _ = particle_states.shape
        logw = torch.empty((N, M), device=self.drone_vec.device)
        Hf, Wf = self.mapF.shape[-2:]

        simmap = compute_heatmap_ncc(self.mapF, self.drone_vec)

        H_feat, W_feat = simmap.shape

        best_i, best_j = np.unravel_index(simmap.argmax(), simmap.shape)
        sim = simmap[best_i][best_j] 



        for b in range(N):
            for m in range(M):
                # 1. Convert (x, y) to pixel coordinates in ortho crop
                x, y = particle_states[b, m].tolist()

                # TODO
                px, py = world_to_heatmap(self.transform, Wf, Hf, W_feat, H_feat, self.drone_vec.shape[2], x, y) #get pixel locations on the heat map
                sim = simmap[py][px] 

                logw[b, m] = float(sim) / self.temperature
        return logw

    