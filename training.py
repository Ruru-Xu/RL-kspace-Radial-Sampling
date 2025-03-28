from pathlib import Path
import logging
import time
import os
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from rl.utils import eval_mode, set_seed_everywhere
from collections import deque, defaultdict
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from rl.acdc_env_ssim import ACDC_Env
from data_loading.mr_datamodule import MRDataModule
import hydra
import numpy as np
import torch
import random
import warnings
warnings.filterwarnings('ignore')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_data(config):
    return MRDataModule(config=config)

def randomize_seed():
    seed = int(time.time()) % (2**32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(version_base=None, config_path='configs', config_name='train_acdc')
def main(cfg):
    print(f"Current working directory : {os.getcwd()}")
    print(f"hydra path:{HydraConfig.get().run.dir}")
    run_dir = Path(HydraConfig.get().run.dir)
    cfg.snapshot_dir = (run_dir / Path("models")).resolve()
    cfg.env.batch_size = cfg.num_envs

    data_module = get_data(cfg.env)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    logging.debug(f"-----length train_loader:{len(train_loader)} val_loader:{len(val_loader)}-----")

    set_seed_everywhere(cfg.seed)
    num_envs = cfg.num_envs
    num_steps = cfg.num_steps
    ppo_batch_size = int(num_envs * num_steps)
    cfg.ppo_batch_size = ppo_batch_size

    import wandb
    wandb.init(
        project=cfg.project_name,
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=run_dir,

    )

    envs = prepare_train_envs(cfg, train_loader)
    eval_envs = prepare_evaluate_envs(cfg, val_loader)

    # seg_model = load_5segmodels()
    # envs.seg_model = seg_model
    # eval_envs.seg_model = seg_model

    ac = hydra.utils.instantiate(cfg.model, action_space=envs.action_space)
    ac.to(cfg.device)

    writer = SummaryWriter(f"{run_dir}/tb")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    global global_step
    global_step = 0

    load_snapshot(ac, cfg.load_from_snapshot_base_dir)
    train(cfg, ac, envs, eval_envs, writer)


def train(cfg, ac, envs, eval_envs, writer):
    parameters = filter(lambda p: p.requires_grad, ac.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)

    num_envs = cfg.num_envs
    num_steps = cfg.num_steps
    ppo_batch_size = cfg.ppo_batch_size
    minibatch_size = int(cfg.ppo_batch_size // cfg.num_minibatches)

    global global_step
    best_metric = {"ssim": 0.0}
    device = torch.device(cfg.device)
    kmask_shape = (envs.act_dim,)

    print(f"num_steps:{num_steps}, num_envs:{num_envs}")
    obs = torch.zeros((num_steps, num_envs) + envs.observation_space, dtype=torch.complex64).to(device)
    obs_kmask = torch.zeros((num_steps, num_envs) + kmask_shape, dtype=torch.bool).to(device)

    actions = torch.zeros((num_steps, num_envs)).to(device)
    obs_mt = torch.zeros((num_steps, num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    print(f"obs shape:{obs.shape}")
    episode_return = 0
    episode_returns = deque(maxlen=200)

    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = cfg.total_timesteps // ppo_batch_size
    next_obs_mt = torch.tensor(envs.get_remain_epi_lines(), dtype=torch.long).to(device)

    for update in range(1, num_updates + 1):
        randomize_seed()
        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done
            obs_mt[step] = next_obs_mt
            logging.debug(f'step{step}, obs_mt:{next_obs_mt}')

            with torch.no_grad():
                cur_mask = envs.get_cur_mask_2d()
                input_dict = {"kspace": next_obs, "mt": next_obs_mt}
                action, logprob, _, value = ac.get_action_and_value(input_dict, cur_mask)
                values[step] = value

            actions[step] = action
            logprobs[step] = logprob
            obs_kmask[step] = cur_mask.to(device)
            next_obs, reward, done, info = envs.step(action)
            next_obs_mt = torch.tensor(envs.get_remain_epi_lines()).to(device)
            episode_return += reward
            # logging.debug(f'update:{update}, step:{step}, reward:{reward[0]}, done :{done[0]}')
            rewards[step].copy_(reward).to(device).view(-1)
            next_obs, next_done = next_obs.to(device), torch.Tensor(done).to(device)

            if done.item() == 1:
                episode_returns.append(episode_return.cpu().numpy())
                episode_return = 0

        with torch.no_grad():
            print('acc:', 256*256/(torch.count_nonzero(obs[-1])/obs.shape[1]))
            next_value = ac.get_value({'kspace': next_obs, "mt": next_obs_mt}).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * \
                        nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * \
                                             cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.observation_space)
        b_obs_kmask = obs_kmask.reshape((-1,) + kmask_shape)

        b_obs_mt = obs_mt.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(ppo_batch_size)
        clipfracs = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, ppo_batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                input_dict = {"kspace": b_obs[mb_inds], "mt": b_obs_mt[mb_inds]}
                _, newlogprob, entropy, newvalue = ac.get_action_and_value(input_dict, b_obs_kmask[mb_inds], a=b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                           torch.clamp(
                               ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                             ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()

            if cfg.target_kl is not None:
                if approx_kl > cfg.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
                                                  np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step /
                                            (time.time() - start_time)), global_step)

        if update % cfg.log_interval == 0 and len(episode_returns) > 1:
            print(f'[TRAIN] Update: {update}, FPS: {int(global_step / (time.time() - start_time))}, Mean Ret: {np.mean(episode_returns)}')
            writer.add_scalar("charts/train_mean_return", np.mean(episode_returns), global_step)

            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Update {update}, Learning Rate: {current_lr:.6f}")

            episode_return = 0
            episode_returns = deque(maxlen=200)

        # if update % cfg.eval_interval == 0:
        evaluate(ac, eval_envs, writer, best_metric=best_metric, snapshot_dir=cfg.snapshot_dir)

def evaluate(ac, envs, writer=None, best_metric={"ssim": 0.0}, snapshot_dir=None, device='cuda'):
    global global_step
    avg_ssim_scores = []
    avg_returns = deque(maxlen=10000)
    envs.factory_reset()
    obs = envs.reset()
    episode_reward = 0
    obs_mt = torch.tensor(envs.get_remain_epi_lines()).to(device)
    num_done = 0
    step = 0

    while True:
        step += 1
        with torch.no_grad(), eval_mode(ac):
            cur_mask = envs.get_cur_mask_2d()
            input_dict = {"kspace": obs, 'mt': obs_mt}
            action, _, _, _ = ac.get_action_and_value(input_dict, cur_mask, deterministic=True)
            # print(action)
        with torch.no_grad():
            obs, reward, done, info = envs.step(action)
            obs_mt = torch.tensor(envs.get_remain_epi_lines()).to(device)

        episode_reward += reward
        if done.item() == 1:
            avg_returns.append(episode_reward.mean().item())
            avg_ssim_scores.append(info.get('ssim_score', 0.0).mean().item())
            episode_reward = 0

            num_done += 1
            logging.debug(f"Final mask after {num_done} episodes: {info.get('final_mask')}")

            if num_done == len(envs.data_loader):
                break

    # Calculate average SSIM score for the evaluation
    avg_ssim_score = np.mean(avg_ssim_scores)
    print(f'[EVAL] Total episodes: {len(avg_returns)}, Avg Return: {np.mean(avg_returns)}, Avg SSIM: {avg_ssim_score}')

    if writer is not None:
        writer.add_scalar("charts/eval_mean_return", np.mean(avg_returns), global_step)
        writer.add_scalar("charts/eval_mean_ssim", avg_ssim_score, global_step)

        if avg_ssim_score >= best_metric['ssim']:
            best_metric['ssim'] = avg_ssim_score
            logging.info(f"[EVAL] Global Step: {global_step}, Best SSIM: {avg_ssim_score}")
            save_snapshot(ac, snapshot_dir)

        writer.add_scalar("charts/best_ssim", best_metric['ssim'], global_step)
        save_snapshot(ac, snapshot_dir, cur_is_best=False, save_last=True)

def save_snapshot(model, snapshot_dir, save_last=False, cur_is_best=True):
    if snapshot_dir is None:
        return

    logging.info(f"[Train.py] save at snapshot_dir:{snapshot_dir}")
    snapshot_dir.mkdir(exist_ok=True, parents=True)
    if cur_is_best:
        snapshot = snapshot_dir / f'best_model.pt'
        with open(snapshot, 'wb') as f:
            torch.save(model.state_dict(), f)
    if save_last:
        snapshot = snapshot_dir / f'last_model.pt'
        with open(snapshot, 'wb') as f:
            torch.save(model.state_dict(), f)


def load_snapshot(model, load_from_snapshot_base_dir):
    snapshot_base_dir = Path(load_from_snapshot_base_dir)
    snapshot = snapshot_base_dir / f'best_model.pt'
    if not snapshot.exists():
        return None
    logging.info(f"[Train.py] load snapshot:{snapshot}")
    model.load_state_dict(torch.load(snapshot))


def prepare_train_envs(cfg, train_loader):
    observation_space = tuple(cfg.env.observation_space)
    print(cfg.env)
    envs = ACDC_Env(train_loader, budget=cfg.budget, observation_space=observation_space, device=cfg.device)
    return envs


def prepare_evaluate_envs(cfg, val_loader):
    observation_space = cfg.env.observation_space
    envs = ACDC_Env(val_loader, budget=cfg.budget, observation_space=observation_space, device=cfg.device)
    return envs


if __name__ == "__main__":
    main()
