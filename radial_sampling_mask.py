import numpy as np
import torch
from math import pi, cos, sin
from typing import Tuple, List


class RadialSampler:
    def __init__(self, image_size=256, lines_per_action=1):
        self.image_size = image_size
        self.lines_per_action = lines_per_action  # Number of lines to sample per action
        self.golden_angle = 137.508 / 2  # Adjust for 180° space
        self.sampling_radius = image_size // 2.5
        self.center = (image_size // 2, image_size // 2)

        # Initialize importance map for k-space center weighting
        self.importance_map = self._create_importance_map()
        self.sampled_angles = set()  # Track sampled angles

    def _create_importance_map(self):
        """Create importance weighting map for k-space center"""
        y, x = np.ogrid[-self.center[0]:self.center[0],
               -self.center[1]:self.center[1]]
        distance = np.sqrt(x * x + y * y)
        return np.exp(-distance / (self.image_size / 4))

    def _dda_line(self, start, end):
        """Enhanced DDA line algorithm with anti-aliasing"""
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))

        if steps == 0:
            return np.array([y0], dtype=int), np.array([x0], dtype=int)

        # Use numpy for vectorized operations
        t = np.linspace(0, 1, int(steps) + 1)
        x = x0 + dx * t
        y = y0 + dy * t

        return np.round(y).astype(int), np.round(x).astype(int)

    def _create_block_mask(self, center_angle):
        """Create binary mask for multiple adjacent lines"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        for i in range(self.lines_per_action):
            # Calculate angle for each line in the block
            angle = (center_angle + i * (180 / (self.lines_per_action * 4))) % 180
            angle_360 = angle * 2  # Convert to 360° space for calculations
            theta = np.deg2rad(angle_360)
            # max_radius = int(np.hypot(self.image_size, self.image_size) // 2)
            max_radius = int(np.ceil(np.sqrt(2) * self.image_size / 2))

            # Calculate line endpoints
            x_start = int(self.center[0] + max_radius * cos(theta + pi))
            y_start = int(self.center[1] + max_radius * sin(theta + pi))
            x_end = int(self.center[0] + max_radius * cos(theta))
            y_end = int(self.center[1] + max_radius * sin(theta))

            # Generate line coordinates
            rr, cc = self._dda_line((x_start, y_start), (x_end, y_end))

            # Apply validity constraints
            valid = (rr >= 0) & (rr < self.image_size) & (cc >= 0) & (cc < self.image_size)
            rr_valid = rr[valid]
            cc_valid = cc[valid]

            # Apply circular constraint with binary values
            in_circle = ((rr_valid - self.center[1]) ** 2 +
                         (cc_valid - self.center[0]) ** 2 <=
                         self.sampling_radius ** 2)

            # Set valid points to 1 (binary mask)
            mask[rr_valid[in_circle], cc_valid[in_circle]] = 1.0

            self.sampled_angles.add(angle)

        return mask

    def batch_radial_mask(self, action_indices):
        """
        Batch-generate radial masks with block sampling
        Args:
            action_indices: Tensor of shape [batch_size] containing angle indices
        Returns:
            batch_mask: Tensor of shape [batch_size, 1, H, W]
        """
        indices_np = action_indices.cpu().numpy()
        masks = []

        for idx in indices_np:
            # Calculate center angle using golden ratio
            center_angle = (idx * self.golden_angle) % 180

            # Generate block mask
            mask = self._create_block_mask(center_angle)
            masks.append(mask)

        return torch.stack([torch.tensor(m, device="cuda") for m in masks], dim=0).unsqueeze(1)

    def createRadialSampling(self, action_index):
        """Legacy single-mask interface"""
        return self.batch_radial_mask(action_index).squeeze(0)

    def reset(self):
        """Reset sampler state"""
        self.sampled_angles.clear()


def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):
    """Enhanced trajectory generation with block sampling"""
    N_tot_spokes = N_spokes * N_time
    N_samples = base_res * 2

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5 ** 0.5)
    base_rad = np.pi / (2 * (gind + tau - 1))  # Adjust for 180° space

    # Generate angles with golden ratio spacing
    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    # Create trajectory
    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    # Scale and reshape
    traj = traj / 2
    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    return np.squeeze(traj)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test the enhanced sampler
    sampler = RadialSampler(image_size=256, lines_per_action=2)

    # Generate multiple masks to show progression
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i, action in enumerate([0, 1, 2, 3]):
        mask = sampler.createRadialSampling(torch.tensor([action], device="cuda"))
        ax = axes[i // 2, i % 2]
        ax.imshow(mask.cpu().numpy()[0], cmap='gray')
        ax.set_title(f'Action {action}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()