
"""An example of training PPO against OpenAI Gym Atari Envs.
This script is an example of training a PPO agent on Atari envs.
To train PPO for 10M timesteps on Breakout, run:
    python train_ppo_ale.py
To train PPO using a recurrent model on a flickering Atari env, run:
    python train_ppo_ale.py --recurrent --flicker --no-frame-stack
"""
import argparse
import os

import gym
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead, GaussianHeadWithFixedCovariance
from pfrl.wrappers import atari_wrappers
#torch.multiprocessing.set_start_method('spawn')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="InvertedPendulum-v2", help="Gym Env ID."
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device ID. Set to -1 to use CPUs only."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=6,
        help="Number of env instances run in parallel.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps", type=int, default=50 ** 7, help="Total time steps for training."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval (in timesteps) between evaluation phases.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes ran in an evaluation phase.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run demo episodes, not training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Directory path to load a saved agent data from"
            " if it is a non-empty string."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=128 * 8,
        help="Interval (in timesteps) between PPO iterations.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32 * 8,
        help="Size of minibatch (in timesteps).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs used for each PPO iteration.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        help="Interval (in timesteps) of printing logs.",
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        default=False,
        help="Use a recurrent model. See the code for the model definition.",
    )
    parser.add_argument(
        "--flicker",
        action="store_true",
        default=False,
        help=(
            "Use so-called flickering Atari, where each"
            " screen is blacked out with probability 0.5."
        ),
    )
    parser.add_argument(
        "--no-frame-stack",
        action="store_true",
        default=False,
        help=(
            "Disable frame stacking so that the agent can only see the current screen."
        ),
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Frequency at which agents are stored.",
    )

    def str2bool(string):
        string = string.lower()
        trues = ["true", "1", "yes"]
        falses = ["false", "0", "no"]

        if string in trues:
            return True

        if string in falses:
            return False

        raise AssertionError

    parser.add_argument(
        "--diayn-use",
        type=str2bool,
        default=False,
        help="Wether or not we should use diayn",
    )
    parser.add_argument(
        "--diayn-n-skills",
        type=int,
        default=50,
        help="Number of skills to train",
    )

    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(idx, test, discriminator):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)

        env = pfrl.wrappers.CastObservationToFloat32(env)

        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if not test and not args.diayn_use:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, 1e-2)  # todo add to args https://github.com/pfnet/pfrl/blob/44bf2e483f5a2f30be7fd062545de306247699a1/examples/gym/train_reinforce_gym.py#L45
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test, discriminator, force_no_diayn=False):  # force_no_diayn is used to compte the obs space for DIAYN
        env_funs = []
        for idx, env in enumerate(range(args.num_envs)):
            def _temp():
                return make_env(idx, test, discriminator)
            env_funs.append(_temp)


        vec_env = pfrl.envs.MultiprocessVectorEnv(
            env_funs
        )

        if not force_no_diayn and args.diayn_use:
            from diayn_sim import DIAYNWrapper
            vec_env = DIAYNWrapper(vec_env, discriminator, args.diayn_n_skills)

        return vec_env

    discriminator = None
    if args.diayn_use:
        from diayn_sim import Discriminator

        sample_env = make_batch_env(test=False, discriminator=None, force_no_diayn=True)
        obs_space = sample_env.observation_space.shape
        print("Old observation space", sample_env.observation_space)
        print("Old action space", sample_env.action_space)
        del sample_env

        discriminator = Discriminator(
            input_size=obs_space,
            layers=(200,200),
            n_skills=args.diayn_n_skills
        ).cuda()

    sample_env = make_batch_env(test=False, discriminator=discriminator)
    print("Observation space", sample_env.observation_space)
    print("Action space", sample_env.action_space)
    action_space = sample_env.action_space
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    obs_size = obs_space.low.size
    hidden_size = 200   # https://github.com/pfnet/pfrl/blob/44bf2e483f5a2f30be7fd062545de306247699a1/examples/gym/train_reinforce_gym.py#L84
    del sample_env

    if isinstance(action_space, gym.spaces.Box):
        model = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            pfrl.nn.Branched(
                nn.Sequential(
                    nn.Linear(hidden_size, action_space.low.size),
                    GaussianHeadWithFixedCovariance(0.3),
                ),
                nn.Linear(hidden_size, 1),
            ),
        )
    else:
        model = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, action_space.n),
            SoftmaxCategoricalHead(),
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        update_interval=args.update_interval,
        minibatch_size=args.batchsize,
        epochs=args.epochs,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-2,
        recurrent=args.recurrent,
        max_grad_norm=0.5,
    )
    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)
        )

        eval_hooks = []
        if args.diayn_use:
            def log_disc(env, agent, evaluator, t, eval_score):
                env.call_logging()
            eval_hooks.append(log_disc)


        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False, discriminator),
            eval_env=make_batch_env(True, discriminator),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            checkpoint_freq=args.checkpoint_frequency,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_best_so_far_agent=False,
            step_hooks=step_hooks,
            evaluation_hooks=eval_hooks
        )


if __name__ == "__main__":
    main()