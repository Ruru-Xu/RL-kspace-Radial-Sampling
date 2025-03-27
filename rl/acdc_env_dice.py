import fastmri
import torch
import numpy as np
from pytorch_msssim import ssim as py_msssim
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity

from segment.seg_test import get_seg_result
from radial_sampling_mask import RadialSampler
radial_sampler = RadialSampler()

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ACDC_Env:

    def __init__(self, data_loader, budget, observation_space=(3, 640, 400), device='cuda'):

        # Environment properties and initialization
        self.state = 0
        self.done = False
        self.data_loader = data_loader
        self.data_loader_iter = iter(self.data_loader)
        self.observation_space = observation_space
        self.action_space = Namespace(n=180)
        self.act_dim = self.action_space.n
        self.device = device
        self.budget = budget
        self.radial_sampler = radial_sampler

    def factory_reset(self):
        self.data_loader_iter = iter(self.data_loader)

    def reset(self):
        # Reset data iterator if needed
        try:
            batch = next(self.data_loader_iter)
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            batch = next(self.data_loader_iter)

        # Move batch data to the designated device
        batch["kspace_fully"] = batch["kspace_fully"].to(torch.complex64).unsqueeze(1).to(self.device)
        batch["target"] = batch["target"].to(self.device)
        self.state = batch

        # Set up initial k-space and accumulated mask
        kspace = batch["kspace_fully"]
        batch_size = kspace.shape[0]
        num_cols = kspace.shape[-1]
        self.num_cols = num_cols
        self.accumulated_mask = batch["initial_mask"].float().unsqueeze(1).to(self.device)
        self.counter = 0

        # Return the masked k-space as initial observation
        s0 = kspace * self.accumulated_mask
        self.done = torch.zeros(batch_size)

        initial_recons = fastmri.complex_abs(fastmri.ifft2c(torch.stack((s0.real, s0.imag), dim=-1)))
        initial_recons = (initial_recons - initial_recons.amin(dim=(-1, -2), keepdim=True)) / (
                    initial_recons.amax(dim=(-1, -2), keepdim=True) - initial_recons.amin(dim=(-1, -2), keepdim=True))

        heart_mask = self.state['seg_mask'].to(self.device)
        self.previous_heart_ssim, self.previous_heart_mse = self.calculate_heart_ssim_mse(initial_recons, self.state['target'].unsqueeze(1), heart_mask)
        self.previous_global_ssim = py_msssim(self.state['target'].unsqueeze(1), initial_recons, data_range=1.0,
                                              size_average=False)
        # self.previous_dice_score = get_seg_result(initial_recons.squeeze(1).cpu().detach().numpy(), self.seg_model, self.state['seg_mask'].numpy())
        return s0


    def get_state(self):
        return self.state

    def get_reward(self, observation):
        recons = fastmri.complex_abs(fastmri.ifft2c(torch.stack((observation.real, observation.imag), dim=-1)))
        recons = (recons - recons.amin(dim=(-1, -2), keepdim=True)) / (
                recons.amax(dim=(-1, -2), keepdim=True) - recons.amin(dim=(-1, -2), keepdim=True))
        global_ssim = py_msssim(self.state['target'].unsqueeze(1), recons, data_range=1.0, size_average=False)
        heart_mask = self.state['seg_mask'].to(self.device)

        # Calculate heart-focused metrics
        heart_ssim, heart_mse = self.calculate_heart_ssim_mse(recons, self.state['target'].unsqueeze(1), heart_mask)

        # dice_score = torch.tensor(get_seg_result(recons.squeeze(1).cpu().detach().numpy(), self.seg_model, self.state['seg_mask'].numpy()), device=self.device)

        delta_global_ssim = global_ssim - self.previous_global_ssim
        delta_heart_ssim = heart_ssim - self.previous_heart_ssim
        delta_heart_mse = heart_mse - self.previous_heart_mse
        # delta_dice_score = dice_score - self.previous_dice_score

        alpha, beta = calculate_dynamic_weight(global_ssim.mean().item(), heart_ssim.mean().item(), self.counter, self.budget)
        reward = 0.6 * (alpha * delta_global_ssim + beta * delta_heart_ssim) - 0.4 * delta_heart_mse

        # Update previous scores
        self.previous_heart_ssim = heart_ssim
        # self.previous_dice_score = dice_score
        self.previous_global_ssim = global_ssim
        self.previous_heart_mse = heart_mse

        return reward


    def get_cur_mask_2d(self):
        if torch.sum(self.accumulated_mask[0]) == 1024:
            self.cur_mask = torch.ones((self.accumulated_mask.shape[0], 180), dtype=torch.bool, device=self.accumulated_mask.device)
            return self.cur_mask
        else:
            return self.cur_mask

    def get_remain_epi_lines(self):
        return self.budget - self.counter

    def set_budget(self, num_lines):
        self.budget = num_lines

    def reach_budget(self):
        return self.counter >= self.budget

    def get_accumulated_mask(self):
        return self.accumulated_mask

    def step(self, action):
        info = {}

        batch_indices = torch.arange(action.size(0), device=action.device)
        self.cur_mask[batch_indices, action] = False

        # Update accumulated mask
        radial_mask = self.radial_sampler.createRadialSampling(action)
        # Update accumulated mask
        self.accumulated_mask = torch.max(self.accumulated_mask, radial_mask)

        self.counter += 1

        # Get observation and reward
        state = self.get_state()
        observation = state['kspace_fully'] * self.accumulated_mask

        reward = self.get_reward(observation)

        if self.reach_budget():
            done = torch.ones(1)  # Single done flag as budget is reached
            info['ssim_score'] = self.previous_heart_ssim  # Optionally log SSIM score
            info['final_mask'] = self.accumulated_mask.clone().cpu().numpy()  # Optionally log mask
            observation = self.reset()
        else:
            done = torch.zeros(1)  # Single done flag as budget not yet reached

        return observation, reward, done, info

    def calculate_heart_ssim_mse(self, recons, target, heart_mask):
        """Calculate SSIM only for the heart region"""
        # Get dimensions
        B, C, H, W = recons.shape

        heart_ssim_values = []
        heart_mses = []
        for b in range(B):
            # Get mask indices
            mask = heart_mask[b]
            mask_indices = torch.where(mask > 0)

            if len(mask_indices[0]) == 0:  # Skip if no heart region
                heart_ssim_values.append(torch.tensor(0.0, device=self.device))
                continue

            # Extract heart region only
            min_h, max_h = mask_indices[0].min(), mask_indices[0].max()
            min_w, max_w = mask_indices[1].min(), mask_indices[1].max()

            # Crop to heart region
            recons_heart_region = recons[b, :, min_h:max_h + 1, min_w:max_w + 1]
            target_heart_region = target[b, :, min_h:max_h + 1, min_w:max_w + 1]
            mask_heart_region = heart_mask[b, min_h:max_h + 1, min_w:max_w + 1]

            # Apply mask
            recons_heart_region = recons_heart_region * mask_heart_region
            target_heart_region = target_heart_region * mask_heart_region

            # Calculate SSIM only for heart region
            ssim_val = py_msssim(target_heart_region.unsqueeze(0), recons_heart_region.unsqueeze(0), data_range=1.0, size_average=True, win_size=3)
            heart_ssim_values.append(ssim_val)

            # Calculate MSE in image domain
            mse = torch.mean((recons_heart_region - target_heart_region) ** 2)
            mse_normalized = mse / (target_heart_region.max() ** 2)
            heart_mses.append(mse_normalized)

        return torch.stack(heart_ssim_values), torch.stack(heart_mses)

def calculate_dynamic_weight(ssim_global, ssim_lesion, step, total_steps):
    diff = abs(ssim_global - ssim_lesion)
    progress = step / total_steps  # 从0到1
    # 当全局与病灶差异很小，则保持平衡；否则随进度提升对病灶的关注
    if diff < 0.01:
        alpha = 0.4
        beta = 0.6
    else:
        # 从 0.3 ~ 0.1 之间变化，表示全局比重降低
        alpha = max(0.1, 0.4 - 0.3 * progress)
        beta = 1.0 - alpha
    return alpha, beta

if __name__ == "__main__":
    pass
