import random
import time
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from shimmy import MeltingPotCompatibilityV0
import supersuit as ss
from default_args import parse_args

from record_ma_episode_statistics import (
    RecordMultiagentEpisodeStatistics,
)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, agent_id):
        super().__init__()
        # 1 frame
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        # x here is obs: (B, 88, 88, 3)
        # Convert to tensor, rescale to [0, 1]
        x = (x - 10.0) / (255.0 - 10.0)
        #   B x H x W x C to B x C x H x W
        return self.critic(self.network(x.permute(0, 3, 1, 2)))

    def get_action_and_value(self, x, action=None):
        # x here is [B, 88, 88, 3]
        x = (x - 10.0) / (255.0 - 10.0)
        hidden = self.network(x.permute(0, 3, 1, 2))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class MultiAgents(nn.Module):
    def __init__(self, envs, num_agents):
        super().__init__()
        self.num_agents = num_agents
        # 1 frame
        self.networks = nn.ModuleList(
            Agent(envs, agent_id).to(device) for agent_id in range(self.num_agents)
        )

    def get_values(self, x):
        # x herer is obs: (16, 88, 88, 3)
        values = [self.networks[a].get_value(x[a].unsqueeze(0)) for a in range(self.num_agents)]
        values = torch.cat(values, dim=0)
        return values

    def get_actions_and_values(self, x, given_actions=None):
        actions, log_probs, entropies, values = [], [], [], []
        for a in range(self.num_agents):
            if given_actions is None:
                # x herer is obs: (1, 16, 88, 88, 3) -> (1)
                action, log_prob, entropy, value = self.networks[a].get_action_and_value(x[:, a])
            else:
                # x herer is obs: (128, 16, 88, 88, 3), given_actions (128, 16) -> (128)
                action, log_prob, entropy, value = self.networks[a].get_action_and_value(
                    x[:, a], action=given_actions[:, a]
                )
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
        actions = torch.cat(actions, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        entropies = torch.cat(entropies, dim=0)
        values = torch.cat(values, dim=0)
        return actions, log_probs, entropies, values


def unbatchify(x, possible_agents):
    x = x.detach().cpu().numpy()
    x = {a: x[i] for i, a in enumerate(possible_agents)}
    return x


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.tensorboard.patch(root_logdir=f"results/{run_name}", pytorch=True)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"results/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # run on M2 chip for development; cuda for training
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "mps")

    # ENV setup - skip helper wrappers like colour reduction, frame stack on atari envs for now
    envs = MeltingPotCompatibilityV0(substrate_name=args.env_id, render_mode="rgb_array")
    num_agents = envs._num_players
    possible_agents = envs.possible_agents  # dict
    envs = ss.pettingzoo_env_to_vec_env_v1(envs)
    envs = ss.concat_vec_envs_v1(
        envs,
        num_vec_envs=1,
        num_cpus=0,
        base_class="gymnasium",
    )
    # all agents have the same obs & action space in substrates we run on
    envs.single_observation_space = envs.observation_space["RGB"]  # if vec env
    envs.single_action_space = envs.action_space  # if vec env
    envs.is_vector_env = True
    envs.render_mode = "rgb_array"  # supersuit wrapper makes property gone
    envs = RecordMultiagentEpisodeStatistics(envs, num_agents)
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")

    agents = MultiAgents(envs, num_agents).to(device)
    optimizer = optim.Adam(agents.parameters(), lr=args.learning_rate, eps=1e-5)
    print(agents)

    # ALGO logic: Storage setup
    # remove all num_envs dimension as not considering vec_env for now
    # (512, 16, 88, 88, 3)
    obs = torch.zeros((args.num_steps, num_agents) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, num_agents) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, num_agents)).to(device)
    terminations = torch.zeros((args.num_steps, num_agents)).to(device)
    truncations = torch.zeros((args.num_steps, num_agents)).to(device)
    values = torch.zeros((args.num_steps, num_agents)).to(device)

    # TRAINING LOOP
    global_step = 0
    start_time = time.time()
    next_obs, _info = envs.reset()  # seeding not need as Melting Pot is non-determistic
    # TODO: need other obs as well
    next_obs = torch.Tensor(next_obs["RGB"]).to(device)
    next_termination = torch.zeros(num_agents).to(device)
    next_truncation = torch.zeros(num_agents).to(device)
    num_updates = args.total_timesteps // args.batch_size
    print(
        f"num_updates = total_timesteps / batch_size: {args.total_timesteps} / {args.batch_size} = {num_updates}"
    )

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            terminations[step] = next_termination
            truncations[step] = next_truncation

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agents.get_actions_and_values(next_obs.unsqueeze(0))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data
            next_obs, reward, termination, truncation, info = envs.step(action.cpu().numpy())

            # (16, )
            rewards[step] = torch.tensor(reward).to(device)
            next_termination = torch.tensor(termination).to(device)
            next_truncation = torch.tensor(truncation).to(device)
            next_obs = torch.Tensor(next_obs["RGB"]).to(device)

            # for idx, item in enumerate(info):
            #     player_idx = idx % num_agents
            #     if "episode" in item.keys():
            #         print(
            #             f"global_step={global_step}, {player_idx}-episodic_return={item['episode']['r']}"
            #         )
            #         writer.add_scalar(
            #             f"charts/episodic_return-player{player_idx}",
            #             item["episode"]["r"],
            #             global_step,
            #         )

            # 4 social outcome metrics from RecordMultiagentEpisodeStatistics
            if "ma_episode" in info[0].keys():
                print(
                    f"global_step={global_step}, episodic_max_length={info[0]['ma_episode']['l']}"
                )
                print(
                    f"global_step={global_step}, episodic_efficiency={info[0]['ma_episode']['u']}"
                )
                print(f"global_step={global_step}, episodic_equality={info[0]['ma_episode']['e']}")
                print(
                    f"global_step={global_step}, episodic_sustainability={info[0]['ma_episode']['s']}"
                )
                print(f"global_step={global_step}, episodic_peace={info[0]['ma_episode']['s']}")
                if args.track:
                    writer.add_scalar(
                        f"charts/episodic_max_length",
                        info[0]["ma_episode"]["l"],
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/episodic_efficiency",
                        info[0]["ma_episode"]["u"],
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/episodic_equality",
                        info[0]["ma_episode"]["e"],
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/episodic_sustainability",
                        info[0]["ma_episode"]["s"],
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/episodic_peace",
                        info[0]["ma_episode"]["p"],
                        global_step,
                    )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agents.get_values(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            next_done = torch.maximum(next_termination, next_truncation)
            dones = torch.maximum(terminations, truncations)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        # b_logprobs = logprobs.reshape(-1)
        # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # b_advantages = advantages.reshape(-1)
        # b_returns = returns.reshape(-1)
        # b_values = values.reshape(-1)

        # for each agent decentralized obs & actions to learn, no need to flatten, all (batch, num_agents)
        b_obs = obs
        b_logprobs = logprobs
        b_actions = actions
        b_advantages = advantages
        b_returns = returns
        b_values = values

        # Optimizing the policy and value networks for each agent:
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agents.get_actions_and_values(
                    # (B, 16, 88, 88, 3); (B, 16)
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds],
                )
                # dealing with (B, 16) onwards here instead of the flattened batch dim like used to - a bit UGLY
                newlogprob = newlogprob.reshape(-1, num_agents)
                newvalue = newvalue.reshape(-1, num_agents)
                entropy = entropy.reshape(-1, num_agents)

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean(axis=0)
                    approx_kl = ((ratio - 1) - logratio).mean(axis=0)
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean(axis=0)]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean(axis=0)) / (
                        mb_advantages.std(axis=0) + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean(axis=0)

                # Value loss
                # newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean(axis=0)
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean(axis=0)

                entropy_loss = entropy.mean(axis=0)
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # NOTE: Experiment-sum over all agents' loss to 1 element for torch to backprop
                loss = loss.sum()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agents.parameters(), args.max_grad_norm)
                optimizer.step()

            # KL divergence early stopping
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # STATS for the experiment
        agent_clipfracs = torch.stack(clipfracs, dim=0).sum(0)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true, axis=0)
        explained_var = (
            np.nan if (var_y == 0).all() else 1.0 - np.var(y_true - y_pred, axis=0) / var_y
        )

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        for i, agent in enumerate(possible_agents):
            writer.add_scalar(f"losses/value_loss/{agent}", v_loss[i], global_step)
            writer.add_scalar(f"losses/policy_loss/{agent}", pg_loss[i], global_step)
            writer.add_scalar(f"losses/entropy/{agent}", entropy_loss[i], global_step)
            writer.add_scalar(
                f"losses/old_approx_kl/{agent}",
                old_approx_kl[i],
                global_step,
            )
            writer.add_scalar(f"losses/approx_kl/{agent}", approx_kl[i], global_step)
            writer.add_scalar(f"losses/clipfrac/{agent}", agent_clipfracs[i], global_step)
            writer.add_scalar(f"losses/explained_variance/{agent}", explained_var[i], global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
