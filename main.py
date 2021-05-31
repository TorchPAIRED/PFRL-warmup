
"""An example of training PPO against OpenAI Gym Atari Envs.
This script is an example of training a PPO agent on Atari envs.
To train PPO for 10M timesteps on Breakout, run:
    python train_ppo_ale.py
To train PPO using a recurrent model on a flickering Atari env, run:
    python train_ppo_ale.py --recurrent --flicker --no-frame-stack
"""
import argparse

import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.wrappers import atari_wrappers
#torch.multiprocessing.set_start_method('spawn')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="BreakoutNoFrameskip-v4", help="Gym Env ID."
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
        "--steps", type=int, default=10 ** 7, help="Total time steps for training."
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
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test,
            flicker=args.flicker,
            frame_stack=False,
        )
        env.seed(env_seed)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test, discriminator):
        env_funs = []
        for idx, env in enumerate(range(args.num_envs)):
            def _temp():
                return make_env(idx, test, discriminator)
            env_funs.append(_temp)


        vec_env = pfrl.envs.MultiprocessVectorEnv(
            env_funs
        )
        if not args.no_frame_stack:
            vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)

        if args.diayn_use:
            from diayn_sim import DIAYNWrapper
            def augment_obs(obs, z):
                for envs in obs:
                    for frame in envs._frames:
                        #import matplotlib.pyplot as plt
                        #plt.imshow(np.squeeze(frame, axis=0))
                        #plt.show()

                        frame[:, :, 0:4] = z # just in case there is some collapsing going on with too-small values

                        #import matplotlib.pyplot as plt
                        #plt.imshow(np.squeeze(frame, axis=0))
                        #plt.show()

                return obs
            vec_env = DIAYNWrapper(vec_env, discriminator, args.diayn_n_skills, augment_obs_func=augment_obs)


        return vec_env

    discriminator = None
    if args.diayn_use:
        from diayn_sim import Discriminator

        pooler = torch.nn.MaxPool2d(3, )
        def preprocess(input):
            # don't care about framestack since here we only care about the STATES, not trajectory
            # also, we want a batch of vectors, not matrices

            frames = np.array([inpu.__array__() for inpu in input])
            frames = torch.tensor(frames)[:,-1,:,:] # take latest frame only
            frames = pooler(frames.float())

            flattened = torch.flatten(frames, start_dim=1).to(torch.float32)

            flattened -= flattened.min(1, keepdim=True)[0]
            flattened /= flattened.max(1, keepdim=True)[0]

            return flattened

        discriminator = Discriminator(
            input_size=784,
            layers=(300,200),
            n_skills=args.diayn_n_skills,
            preprocess=preprocess
        ).cuda()

    sample_env = make_batch_env(test=False, discriminator=discriminator)
    print("Observation space", sample_env.observation_space)
    print("Action space", sample_env.action_space)
    n_actions = sample_env.action_space.n
    obs_n_channels = sample_env.observation_space.low.shape[0]
    del sample_env

    def lecun_init(layer, gain=1):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pfrl.initializers.init_lecun_normal(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        else:
            pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
            pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
        return layer

    if args.recurrent:
        model = pfrl.nn.RecurrentSequential(
            lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
            nn.ReLU(),
            lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            lecun_init(nn.Linear(3136, 512)),
            nn.ReLU(),
            lecun_init(nn.GRU(num_layers=1, input_size=512, hidden_size=512)),
            pfrl.nn.Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(512, n_actions), 1e-2),
                    SoftmaxCategoricalHead(),
                ),
                lecun_init(nn.Linear(512, 1)),
            ),
        )
    else:
        model = nn.Sequential(
            lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
            nn.ReLU(),
            lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            lecun_init(nn.Linear(3136, 512)),
            nn.ReLU(),
            pfrl.nn.Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(512, n_actions), 1e-2),
                    SoftmaxCategoricalHead(),
                ),
                lecun_init(nn.Linear(512, 1)),
            ),
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        phi=phi,
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