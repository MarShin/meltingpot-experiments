import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import supersuit as ss
import utils
from meltingpot import substrate
from record_ma_episode_statistics import (
    RecordMultiagentEpisodeStatistics,
)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Meltingpot",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="allelopathic_harvest__open",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000, # probably 2MM at least
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=512, # 1000 rollout_len in sb3_train
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps) # 1 * 512
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 512 / 4 = 128
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {f"player_{i}": x[i] for i in range(num_agents)}
    return x


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            # if intput is Linear layer: np.array(envs.single_observation_space.shape).prod()
            # 28 = 4 frames * 3 RGB channels + 16 agent indicator
            layer_init(nn.Conv2d(28, 32, 8, stride=4)),
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
        # x here is obs: (16, 88, 88, 28)
        x = x.clone()
        # Convert to tensor, rescale to [0, 1], and convert from
        # 3 rgb channels * 4 stack frames, rest are agent_indicator
        x[:, :, :, :12] /= 255.0
        #   B x H x W x C to B x C x H x W
        return self.critic(self.network(x.permute(0, 3, 1, 2)))

    def get_action_and_value(self, x, action=None):
        x = x.clone()
        x[:, :, :, :12] /= 255.0
        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_config = substrate.get_config(args.env_id)
    env = utils.parallel_env(
        max_cycles=args.num_steps,
        env_config=env_config,
    )
    num_agents = env.max_num_agents
    env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(
        env, type_only=False
    )  # not added in demo code but most likely useful
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=args.num_envs
        // num_agents,  # number of parallel multi-agent environments
        num_cpus=0,
        base_class="gymnasium",
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    # envs = RecordMultiagentEpisodeStatistics(envs, args.num_steps)

    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(agent)

    # ALGO logic: Storage setup
    # (512, 16, 88, 88, 28)
    obs = torch.zeros(
        # (args.num_steps, args.num_envs)
        (args.num_steps, args.num_envs * num_agents)
        + envs.single_observation_space.shape
    ).to(device)
    # (512, 16)
    actions = torch.zeros(
        (args.num_steps, args.num_envs * num_agents) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs * num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs * num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs * num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs * num_agents)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    _obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(_obs).to(device)
    next_done = torch.zeros(args.num_envs * num_agents).to(device)
    num_updates = args.total_timesteps // args.batch_size
    print(
        f"num_updates = total_timesteps / batch_size: {args.total_timesteps} / {args.batch_size} = {num_updates}"
    )
    print("next_obs shape:", next_obs.shape)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data
            next_obs, reward, terminations, truncations, info = envs.step(
                unbatchify(action, envs)
            )
            rewards[step] = torch.tensor(reward).to(device).view(-1)  # (16, )
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

            for idx, item in enumerate(info):
                player_idx = idx % num_agents
                if "episode" in item.keys():
                    print(
                        f"global_step={global_step}, {player_idx}-episodic_return={item['episode']['r']}"
                    )
                    writer.add_scalar(
                        f"charts/episodic_return-player{player_idx}",
                        item["episode"]["r"],
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/episodic_length-player{player_idx}",
                        item["episode"]["l"],
                        global_step,
                    )

            # episode-wide info - overhead, each tuple in list contains same info
            if "ma_episode" in info[0].keys():
                print(
                    f"global_step={global_step}, multiagent-max_length={info[0]['ma_episode']['l']}"
                )
                print(
                    f"global_step={global_step}, multiagent-episodic_efficiency={info[0]['ma_episode']['u']}"
                )
                print(
                    f"global_step={global_step}, multiagent-episodic_equality={info[0]['ma_episode']['e']}"
                )
                print(
                    f"global_step={global_step}, multiagent-episodic_sustainability={info[0]['ma_episode']['s']}"
                )
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

        # bootstrap value if not done - REVISIT CODE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value networks
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # KL divergence early stopping
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()
