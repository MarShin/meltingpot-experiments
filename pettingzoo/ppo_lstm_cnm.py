import argparse
from operator import ne
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import supersuit as ss
from examples.pettingzoo import utils
from meltingpot.python import substrate
from examples.pettingzoo.record_ma_episode_statistics import (
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
    parser.add_argument("--env-id", type=str, default="allelopathic_harvest", # commons_harvest_open, allelopathic_harvest, clean_up
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000, # probably 2MM at least
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=512,
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

    # Classifier Norm Model (CNM) specific
    parser.add_argument("--pseudo-alpha", type=float, default=0.2,
        help="the pseudoreward term if classifier predicts the disapproval correctly")
    parser.add_argument("--pseudo-beta", type=float, default=0.4,
        help="the pseudoreward term if classifier predicts the disapproval correctly")
    parser.add_argument("--cnm-freeze-step", type=float, default=1e8,
        help="step to freeze after training the classifier")
    parser.add_argument("--cnm-coef", type=float, default=0.01,
        help="coefficient of the classifier CNM loss")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps) # 1 * 512; 
    # not relevant for LSTM training
    # args.minibatch_size = int(args.batch_size // args.num_minibatches) # 512 / 1 = 512
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.network = nn.Sequential(
            # if intput is Linear layer: np.array(envs.single_observation_space_shape).prod()
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

        self.classifier = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, 8, stride=8)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 4, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 32, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 2)),
            nn.Softmax(dim=1),  # no_zap or zap
        )

        # (512 + 2) sanction prediction
        self.lstm = nn.LSTM(514, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_classification(self, obs):
        x = obs.clone()
        # only using T-1 RGB frame and disapproval event for classifier; remove padding
        x = x[:, :, :, [0, 1, 2]] / 255.0
        return self.classifier(x.permute((0, 3, 1, 2)))

    def get_states(self, obs, lstm_state, done, sanction_prediction):
        # x here is obs: (2048, 88, 88, 8) 2 frames stacked
        # Only use the t frame for policy network; t-1 frame for classifier
        x = obs.clone()
        x = x[:, :, :, [4, 5, 6]]
        # Rescale RGB channels to [0, 1]
        x /= 255.0
        # B x H x W x C to B x C x H x W
        hidden = self.network(x.permute((0, 3, 1, 2)))

        # CNM concat (16, 512) with(16, 2)
        hidden = torch.cat((hidden, sanction_prediction), 1)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        # multiply by done flag to reset states to zero during rollout or training
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),  # input
                (  # h0, c0
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done, sanction_prediction):
        hidden, _ = self.get_states(x, lstm_state, done, sanction_prediction)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        sanction_prediction = self.get_classification(x)
        hidden, lstm_state = self.get_states(x, lstm_state, done, sanction_prediction)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            lstm_state,
            sanction_prediction,
        )


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

    def combine_world_obs_fn(obs):
        # sanction-observation function - ways to aggregate the WORLD obs
        # B = J * C * Z WHERE J is sanction opportunity
        # C is context a.k.a last obs, Z is disapproval event 'WHO_ZAPPED_WHO'

        rgb = obs["RGB"]
        who_zap_who = obs["WORLD.WHO_ZAPPED_WHO"]  # label (16, 16)
        avatar_in_range_to_zap = obs[
            "AVATAR_IDS_IN_RANGE_TO_ZAP"
        ]  # J info below (16,) one-hot
        avatar_in_view = obs["AVATAR_IDS_IN_VIEW"]  # (16,) one-hot; no need ATM
        ready_to_shoot = obs["READY_TO_SHOOT"]  # binary float64

        zeros_to_pad = rgb.shape[0] - who_zap_who.shape[0]
        padded_obs = np.pad(
            who_zap_who,
            ((0, zeros_to_pad), (0, zeros_to_pad)),
            "constant",
            constant_values=(2),
        )
        padded_obs[
            who_zap_who.shape[0], : avatar_in_range_to_zap.shape[0]
        ] = avatar_in_range_to_zap
        padded_obs[
            (who_zap_who.shape[0] + 1), : avatar_in_view.shape[0]
        ] = avatar_in_view
        padded_obs[(who_zap_who.shape[0] + 2), 0] = ready_to_shoot

        padded_obs = padded_obs.T.reshape((1, rgb.shape[0], rgb.shape[1]))
        combined_obs = np.concatenate((rgb.T, padded_obs), axis=0).T
        return combined_obs

    def combine_world_obs_space_fn(obs_space):
        # gym.spaces.Dict did not work
        rgb_shape = obs_space["RGB"].shape
        return gym.spaces.Box(0, 255, (*rgb_shape[:2], rgb_shape[-1] + 1), np.uint8)

    env = ss.observation_lambda_v0(
        env,
        lambda a, _: combine_world_obs_fn(a),
        lambda s: combine_world_obs_space_fn(s),
    )
    env = ss.frame_stack_v1(env, 2)  # stack 1 frame instead of 4 as we're using LSTM
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=args.num_envs
        // num_agents,  # number of parallel multi-agent environments
        num_cpus=0,
        base_class="gym",
    )
    envs.single_observation_space = envs.observation_space
    envs.single_observation_space_shape = envs.single_observation_space.shape
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    envs.num_agents = num_agents
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = RecordMultiagentEpisodeStatistics(envs)

    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(agent)

    # ALGO logic: Storage setup
    # (512, 16, 88, 88, 6)
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space_shape
    ).to(device)
    # (512, 16)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    pseudorewards = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # lstm hidden state & cell state
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(
            device
        ),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(
            device
        ),
    )
    num_updates = args.total_timesteps // args.batch_size

    print(
        f"num_updates = total_timesteps / batch_size: {args.total_timesteps} / {args.batch_size} = {num_updates}"
    )
    print("next_obs shape:", next_obs.shape)

    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        sanction_events = []
        sanction_targets = []

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
                (
                    action,
                    logprob,
                    _,
                    value,
                    next_lstm_state,
                    sanction_prediction,
                ) = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)  # (16, )
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

            # Log data for sanction events at time T-1 from (16, 88, 88, 8) stacked frame
            extra_obs = np.squeeze(next_obs[:, :, :, [3]])[:, :19, :16]  # (16, 19, 16)
            obs_avatar_in_range_to_zap = extra_obs[
                :, -3, :
            ]  # (16, 16) - each agent's (16,) one hot
            obs_ready_to_shoot = extra_obs[:, -1, 0]  # (16, 1)

            # if sanction opportunity - save data
            for agent_id in range(num_agents):
                if (
                    obs_ready_to_shoot[agent_id] == 1.0
                    and (obs_avatar_in_range_to_zap[agent_id] == 1.0).any()
                ):
                    # Context C at time T-1
                    sanction_events.append(next_obs[agent_id, :, :, [0, 1, 2]])
                    obs_who_zap_who = np.squeeze(
                        next_obs[agent_id, :, :, [7]][agent_id, :16]
                    )
                    sanction_targets.append(obs_who_zap_who)

            # If no sanction opportunities at all in the episode
            if len(sanction_targets) == 0:
                print("no sanction opportunities at all in episode")
                sanction_targets.append(torch.zeros(16))
                empty_obs = torch.zeros_like(next_obs[agent_id, :, :, [0, 1, 2]])
                sanction_events.append(empty_obs)

            # Compute Pseudorewards of CNM (16, 1) * (16, 1)
            pseudorewards[step] = (
                args.pseudo_alpha * sanction_targets[-1] * sanction_prediction[:, 1]
            ) - args.pseudo_beta * sanction_targets[-1] * sanction_prediction[:, 0]

            # At the end of episode - per agent info
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
                print(
                    f"global_step={global_step}, multiagent-episodic_peace={info[0]['ma_episode']['p']}"
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
                writer.add_scalar(
                    f"charts/episodic_peace",
                    info[0]["ma_episode"]["p"],
                    global_step,
                )

        # TODO: [paper] for learning randomly subsample p=32 for zap event & p=1024 for no_zap event out of at most 1600 samples - sampling of 2 classes something to experiment with; for now just train with everything
        # Label: binary agent zapped or not at time T
        sanction_targets_binary = (
            torch.any((torch.stack(sanction_targets) == 1.0), 1).float().view(-1, 1)
        )
        sanction_events = torch.cat(sanction_events).view(
            -1, *sanction_events[-1].shape
        )
        print(
            f"Fraction of zaps / zap opportunities: {sanction_targets_binary.sum()} / {sanction_targets_binary.shape[0]}={sanction_targets_binary.sum() / sanction_targets_binary.shape[0]}"
        )
        writer.add_scalar(
            f"charts/num_of_sanctions",
            sanction_targets_binary.sum(),
            global_step,
        )
        writer.add_scalar(
            f"charts/num_of_sanctions_opportunities",
            sanction_targets_binary.shape[0],
            global_step,
        )

        # added to env rewards before going through GAE or discounting
        rewards = rewards + pseudorewards

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs, next_lstm_state, next_done, sanction_prediction
            ).reshape(1, -1)
            if args.gae:
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
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value networks
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches  # 16/4
        envinds = np.arange(args.num_envs)  # 1..16
        flatinds = np.arange(args.batch_size).reshape(
            args.num_steps, args.num_envs
        )  # 8192 -> (512, 16)

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[
                    :, mbenvinds
                ].ravel()  # be really careful about the index

                (
                    _,
                    newlogprob,
                    entropy,
                    newvalue,
                    _,
                    _sanction_prediction,
                ) = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (
                        initial_lstm_state[0][:, mbenvinds],
                        initial_lstm_state[1][:, mbenvinds],
                    ),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
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

                # Classifier loss
                bce_loss = nn.BCELoss()
                cnm_loss = bce_loss(
                    agent.get_classification(sanction_events)[:, 1].view(-1, 1),
                    sanction_targets_binary,
                )  # classfier_zap (num_sanction_opportunities, 1) vs

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
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                    + args.cnm_coef * cnm_loss
                )

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
        writer.add_scalar("losses/classifier_loss", cnm_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    env.close()
    writer.close()
