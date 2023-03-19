from agent_icm import ICMAgent
from agent_mimex import MIMExAgent
from agent_rnd import RNDAgent
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

from tensorboardX import SummaryWriter
import wandb

import numpy as np
import copy
import datetime
import os


def main():
    set_seed(args.seed)

    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'atari':
        env = gym.make(env_id)
        input_size = env.observation_space.shape
    elif env_type == 'doom':
        env = gym.make(env_id)
        input_size = env.observation_space['screen'].shape
    else:
        raise NotImplementedError
    output_size = env.action_space.n

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    print(f"Env={env_type}-{env_id}\nObs: {input_size} | Act: {output_size}")

    # creating logging dir
    curr_time = str(datetime.datetime.now())[:19].replace(' ', '_')
    log_name = f'{args.env}_{train_method}_s{args.seed}_{curr_time}'
    log_dir = os.path.join(args.out_dir, log_name)
    os.makedirs(log_dir, exist_ok=False)
    model_dir = os.path.join(log_dir, 'models')
    os.makedirs(model_dir, exist_ok=False)

    # connect to wandb
    config_dict = dict(config['OPTIONS'])
    config_dict['seed'] = args.seed
    wandb.init(
        project='atari-doom',
        entity='toru',
        name=log_name,
        config=config_dict,
        mode=args.wandb_mode
    )

    is_load_model = (args.ckpt_path != '')
    is_render = False

    writer = SummaryWriter(log_dir)

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])
    max_global_step = int(default_config['MaxStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    eta = float(default_config['ETA'])

    # RND-specific constants
    int_gamma = 0.99
    ext_coef = 2.
    int_coef = 1.

    # MIMEx-specific constants
    seq_expl_len = 5
    mask_ratio = 0.7

    clip_grad_norm = float(default_config['ClipGradNorm'])

    reward_rms = RunningMeanStd()

    pre_obs_norm_step = int(default_config['ObsNormStep'])

    if train_method == 'ICM':
        discounted_reward = RewardForwardFilter(gamma)
        obs_rms = RunningMeanStd(shape=(1, 4, 84, 84))
        agent = ICMAgent(
            input_size,
            output_size,
            num_worker,
            num_step,
            gamma,
            lam=lam,
            learning_rate=learning_rate,
            ent_coef=entropy_coef,
            clip_grad_norm=clip_grad_norm,
            epoch=epoch,
            batch_size=batch_size,
            ppo_eps=ppo_eps,
            eta=eta,
            use_cuda=use_cuda,
            use_gae=use_gae,
            use_noisy_net=use_noisy_net
        )
    elif train_method == 'RND':
        discounted_reward = RewardForwardFilter(int_gamma)
        obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        agent = RNDAgent(
            input_size,
            output_size,
            num_worker,
            num_step,
            gamma,
            lam=lam,
            learning_rate=learning_rate,
            ent_coef=entropy_coef,
            clip_grad_norm=clip_grad_norm,
            epoch=epoch,
            batch_size=batch_size,
            ppo_eps=ppo_eps,
            use_cuda=use_cuda,
            use_gae=use_gae,
            use_noisy_net=use_noisy_net
        )
    else:
        discounted_reward = RewardForwardFilter(gamma)
        obs_rms = RunningMeanStd(shape=(1, 4, 84, 84))
        agent = MIMExAgent(
            input_size,
            output_size,
            num_worker,
            num_step,
            gamma,
            lam=lam,
            learning_rate=learning_rate,
            ent_coef=entropy_coef,
            clip_grad_norm=clip_grad_norm,
            epoch=epoch,
            batch_size=batch_size,
            ppo_eps=ppo_eps,
            eta=eta,
            use_cuda=use_cuda,
            use_gae=use_gae,
            use_noisy_net=use_noisy_net
        )

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'doom':
        env_type = DoomEnvironment
    else:
        raise NotImplementedError

    if is_load_model:
        if use_cuda:
            agent.model.load_state_dict(torch.load(args.ckpt_path))
        else:
            agent.model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    steps = 0

    if train_method in ['MIMEx', 'ICM']:
        while steps < pre_obs_norm_step:
            steps += num_worker
            actions = np.random.randint(0, output_size, size=(num_worker,))

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            for parent_conn in parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                next_obs.append(s[:])

        next_obs = np.stack(next_obs)
        obs_rms.update(next_obs)

    elif train_method == 'RND':
        while steps < pre_obs_norm_step:
            steps += num_worker
            actions = np.random.randint(0, output_size, size=(num_worker,))

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            for parent_conn in parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                next_obs.append(s[3, :, :].reshape([1, 84, 84]))

            if len(next_obs) % (num_step * num_worker) == 0:
                next_obs = np.stack(next_obs)
                obs_rms.update(next_obs)

    print('End to initalize...')

    if train_method == 'ICM':
        while global_step < max_global_step:
            total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_policy = \
                [], [], [], [], [], [], [], [], []
            global_step += (num_worker * num_step)
            global_update += 1

            # Step 1. n-step rollout
            for _ in range(num_step):
                actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))

                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action)

                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                for parent_conn in parent_conns:
                    s, r, d, rd, lr = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)
                    log_rewards.append(lr)

                next_states = np.stack(next_states)
                rewards = np.hstack(rewards)
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)

                # total reward = int reward
                intrinsic_reward = agent.compute_intrinsic_reward(
                    (states - obs_rms.mean) / np.sqrt(obs_rms.var),
                    (next_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                    actions)
                sample_i_rall += intrinsic_reward[sample_env_idx]

                total_int_reward.append(intrinsic_reward)
                total_state.append(states)
                total_next_state.append(next_states)
                total_reward.append(rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_values.append(value)
                total_policy.append(policy)

                states = next_states[:, :, :, :]

                sample_rall += log_rewards[sample_env_idx]

                sample_step += 1
                if real_dones[sample_env_idx]:
                    sample_episode += 1
                    writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                    writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                    writer.add_scalar('data/step', sample_step, sample_episode)
                    wandb.log({
                        'reward_per_epi': sample_rall,
                        'step': sample_step
                    }, step=sample_episode)
                    wandb.log({
                        'reward_per_rollout': sample_rall,
                    }, step=global_update)
                    sample_rall = 0
                    sample_step = 0
                    sample_i_rall = 0

            # calculate last next value
            _, value, _ = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))
            total_values.append(value)
            # --------------------------------------------------

            total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_done = np.stack(total_done).transpose()
            total_values = np.stack(total_values).transpose()
            total_logging_policy = torch.stack(total_policy).view(-1, output_size).cpu().numpy()

            # Step 2. calculate intrinsic reward
            # running mean intrinsic reward
            total_int_reward = np.stack(total_int_reward).transpose()
            total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                            total_int_reward.T])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            reward_rms.update_from_moments(mean, std ** 2, count)

            # normalize intrinsic reward
            total_int_reward /= np.sqrt(reward_rms.var)
            int_reward_avg = np.sum(total_int_reward) / num_worker
            writer.add_scalar('data/int_reward_per_epi', int_reward_avg, sample_episode)
            writer.add_scalar('data/int_reward_per_rollout', int_reward_avg, global_update)
            wandb.log({'int_reward_per_epi': int_reward_avg}, step=sample_episode)
            wandb.log({'int_reward_per_rollout': int_reward_avg}, step=global_update)
            # -------------------------------------------------------------------------------------------

            # logging Max action probability
            max_prob = softmax(total_logging_policy).max(1).mean()
            writer.add_scalar('data/max_prob', max_prob, sample_episode)
            wandb.log({'max_prob': max_prob}, step=sample_episode)
            # TODO: log entropy?

            # Step 3. make target and advantage
            target, adv = make_train_data(total_int_reward,
                                        np.zeros_like(total_int_reward),
                                        total_values,
                                        gamma,
                                        num_step,
                                        num_worker)

            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            # -----------------------------------------------

            # Step 5. Training!
            agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                            (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                            target, total_action,
                            adv,
                            total_policy)

            if global_step % (num_worker * num_step * 100) == 0:
                print('Now Global Step :{}'.format(global_step))
                model_path = os.path.join(model_dir, f'step_{global_step}.pth')
                torch.save({
                    'model': agent.model.state_dict(),
                    'icm':agent.icm.state_dict()
                    }, model_path)

    elif train_method == 'MIMEx':
        while global_step < max_global_step:
            total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_policy = \
                [], [], [], [], [], [], [], [], []

            seq_feat_buffer = torch.zeros(
                num_step, num_worker, seq_expl_len, 512, dtype=torch.float, device=agent.device)

            # Step 1. n-step rollout
            for nstep in range(num_step):
                state_feat, actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))

                if global_step == 0 and nstep == 0:
                    # populate buffer
                    seq_feat_buffer[nstep, :, -1] = state_feat

                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action)

                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                for parent_conn in parent_conns:
                    s, r, d, rd, lr = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)
                    log_rewards.append(lr)

                next_states = np.stack(next_states)
                rewards = np.hstack(rewards)
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)

                # implicit left padding
                seq_feat_buffer[nstep, :, 0:-1] = seq_feat_buffer[nstep, :, 1:]
                next_state_tensor = torch.Tensor(
                    (next_states - obs_rms.mean) / np.sqrt(obs_rms.var)).to(agent.device).float()
                next_state_feat = agent.model.forward_with_feat(next_state_tensor)[0]
                seq_feat_buffer[nstep, :, -1] = next_state_feat

                # total reward = int reward
                intrinsic_reward = agent.compute_intrinsic_reward(seq_feat_buffer[nstep])
                sample_i_rall += intrinsic_reward[sample_env_idx]

                total_int_reward.append(intrinsic_reward)
                total_state.append(states)
                total_next_state.append(next_states)
                total_reward.append(rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_values.append(value)
                total_policy.append(policy)

                states = next_states[:, :, :, :]

                sample_rall += log_rewards[sample_env_idx]

                sample_step += 1
                if real_dones[sample_env_idx]:
                    sample_episode += 1
                    writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                    writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                    writer.add_scalar('data/step', sample_step, sample_episode)
                    wandb.log({
                        'reward_per_epi': sample_rall,
                        'step': sample_step
                    }, step=sample_episode)
                    wandb.log({
                        'reward_per_rollout': sample_rall,
                    }, step=global_update)
                    sample_rall = 0
                    sample_step = 0
                    sample_i_rall = 0
                    seq_feat_buffer *= 0

            # calculate last next value
            _, _, value, _ = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))
            total_values.append(value)
            # --------------------------------------------------

            total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_done = np.stack(total_done).transpose()
            total_values = np.stack(total_values).transpose()
            total_logging_policy = torch.stack(total_policy).view(-1, output_size).cpu().numpy()

            # Step 2. calculate intrinsic reward
            # running mean intrinsic reward
            total_int_reward = np.stack(total_int_reward).transpose()
            total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                            total_int_reward.T])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            reward_rms.update_from_moments(mean, std ** 2, count)

            # normalize intrinsic reward
            total_int_reward /= np.sqrt(reward_rms.var)
            int_reward_avg = np.sum(total_int_reward) / num_worker
            writer.add_scalar('data/int_reward_per_epi', int_reward_avg, sample_episode)
            writer.add_scalar('data/int_reward_per_rollout', int_reward_avg, global_update)
            wandb.log({'int_reward_per_epi': int_reward_avg}, step=sample_episode)
            wandb.log({'int_reward_per_rollout': int_reward_avg}, step=global_update)
            # -------------------------------------------------------------------------------------------

            # logging Max action probability
            max_prob = softmax(total_logging_policy).max(1).mean()
            writer.add_scalar('data/max_prob', max_prob, sample_episode)
            wandb.log({'max_prob': max_prob}, step=sample_episode)
            # TODO: log entropy?

            # Step 3. make target and advantage
            target, adv = make_train_data(total_int_reward,
                                        np.zeros_like(total_int_reward),
                                        total_values,
                                        gamma,
                                        num_step,
                                        num_worker)

            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            # -----------------------------------------------

            # Step 5. Training!
            bert_loss = agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                            (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                            target, total_action,
                            adv,
                            total_policy,
                            seq_feat_buffer)
            wandb.log({'bert_loss': bert_loss}, step=global_update)

            global_step += (num_worker * num_step)
            global_update += 1
            if global_step % (num_worker * num_step * 100) == 0:
                print('Now Global Step :{}'.format(global_step))
                model_path = os.path.join(model_dir, f'step_{global_step}.pth')
                torch.save({
                    'model': agent.model.state_dict(),
                    'bert': agent.bert.state_dict()}, model_path)

    elif train_method == 'RND':
        while global_step < max_global_step:
            total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
                [], [], [], [], [], [], [], [], [], [], []
            global_step += (num_worker * num_step)
            global_update += 1

            # Step 1. n-step rollout
            for _ in range(num_step):
                actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action)

                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                for parent_conn in parent_conns:
                    s, r, d, rd, lr = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)
                    log_rewards.append(lr)
                    next_obs.append(s[3, :, :].reshape([1, 84, 84]))

                next_states = np.stack(next_states)
                rewards = np.hstack(rewards)
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)
                next_obs = np.stack(next_obs)

                # total reward = int reward + ext Reward
                intrinsic_reward = agent.compute_intrinsic_reward(
                    ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
                intrinsic_reward = np.hstack(intrinsic_reward)
                sample_i_rall += intrinsic_reward[sample_env_idx]

                total_next_obs.append(next_obs)
                total_int_reward.append(intrinsic_reward)
                total_state.append(states)
                total_reward.append(rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_ext_values.append(value_ext)
                total_int_values.append(value_int)
                total_policy.append(policy)
                total_policy_np.append(policy.cpu().numpy())

                states = next_states[:, :, :, :]

                sample_rall += log_rewards[sample_env_idx]

                sample_step += 1
                if real_dones[sample_env_idx]:
                    sample_episode += 1
                    writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                    writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                    writer.add_scalar('data/step', sample_step, sample_episode)
                    wandb.log({
                        'reward_per_epi': sample_rall,
                        'step': sample_step
                    }, step=sample_episode)
                    wandb.log({
                        'reward_per_rollout': sample_rall,
                    }, step=global_update)
                    sample_rall = 0
                    sample_step = 0
                    sample_i_rall = 0

            # calculate last next value
            _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            # --------------------------------------------------

            total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_reward = np.stack(total_reward).transpose().clip(-1, 1)
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_done = np.stack(total_done).transpose()
            total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
            total_ext_values = np.stack(total_ext_values).transpose()
            total_int_values = np.stack(total_int_values).transpose()
            total_logging_policy = np.vstack(total_policy_np)

            # Step 2. calculate intrinsic reward
            # running mean intrinsic reward
            total_int_reward = np.stack(total_int_reward).transpose()
            total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                            total_int_reward.T])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            reward_rms.update_from_moments(mean, std ** 2, count)

            # normalize intrinsic reward
            total_int_reward /= np.sqrt(reward_rms.var)
            int_reward_avg = np.sum(total_int_reward) / num_worker
            writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
            writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
            wandb.log({'int_reward_per_epi': int_reward_avg}, step=sample_episode)
            wandb.log({'int_reward_per_rollout': int_reward_avg}, step=global_update)
            # -------------------------------------------------------------------------------------------

            # logging Max action probability
            max_prob = softmax(total_logging_policy).max(1).mean()
            writer.add_scalar('data/max_prob', max_prob, sample_episode)
            wandb.log({'max_prob': max_prob}, step=sample_episode)
            # TODO: log entropy?

            # Step 3. make target and advantage
            # extrinsic reward calculate
            ext_target, ext_adv = make_train_data(total_reward,
                                                total_done,
                                                total_ext_values,
                                                gamma,
                                                num_step,
                                                num_worker)

            # intrinsic reward calculate
            # None Episodic
            int_target, int_adv = make_train_data(total_int_reward,
                                                np.zeros_like(total_int_reward),
                                                total_int_values,
                                                int_gamma,
                                                num_step,
                                                num_worker)

            # add ext adv and int adv
            total_adv = int_adv * int_coef + ext_adv * ext_coef
            # -----------------------------------------------

            # Step 4. update obs normalize param
            obs_rms.update(total_next_obs)
            # -----------------------------------------------

            # Step 5. Training!
            agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                            total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                            total_policy)

            if global_step % (num_worker * num_step * 100) == 0:
                print('Now Global Step :{}'.format(global_step))
                model_path = os.path.join(model_dir, f'step_{global_step}.pth')
                torch.save({
                    'model': agent.model.state_dict(),
                    'predictor': agent.rnd.predictor.state_dict(),
                    'target': agent.rnd.target.state_dict()
                    }, model_path)

    wandb.finish()


if __name__ == '__main__':
    main()
