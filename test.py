import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic


def test(rank, args, shared_model, counter, logger):
    console_f = logger.init_console_log_file()

    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    max_score = 0

    start_time = time.time()

    while True:
        if args.max_counter_num != 0 and counter.value > args.max_counter_num:
            if args.save_policy_models:
                logger.save_policy_model(shared_model, counter.value + 1)
            exit(0)
        # monitor counter value
        if counter.value % args.testing_every_counter > 1:
            continue
        counter_value = counter.value
        model.load_state_dict(shared_model.state_dict())

        if args.save_policy_models:
            if counter_value % args.save_policy_models_every <= 5:
                logger.save_policy_model(shared_model, counter_value)

        state = env.reset()
        state = torch.from_numpy(state)
        reward_sum = 0
        done = True

        # a quick hack to prevent the agent from stucking
        # actions = deque(maxlen=100)
        # actions = deque(maxlen=500)
        actions = deque(maxlen=1000)
        episode_length = 0
        episode_count = 0
        episode_rewards_sum = 0
        episode_length_sum = 0
        while True:
            episode_length += 1
            # Sync with the shared model
            with torch.no_grad():
                if done:
                    cx = Variable(torch.zeros(1, 256))
                    hx = Variable(torch.zeros(1, 256))
                else:
                    cx = Variable(cx.data)
                    hx = Variable(hx.data)

                value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
                prob = F.softmax(logit, dim=1)
                action = prob.max(1, keepdim=True)[1].data.numpy()

            state, reward, done, _ = env.step(action[0, 0])
            done = done or episode_length >= args.max_episode_length
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            actions.append(action[0, 0])
            if actions.count(actions[0]) == actions.maxlen:
                done = True

            if done:
                episode_count += 1
                episode_rewards_sum += reward_sum
                episode_length_sum += episode_length
                if episode_count == args.testing_episodes_num:
                    print("Time {}, num steps {}, FPS {:.0f}, avg episode reward {}, avg episode length {}".format(
                        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                        counter_value, counter_value / (time.time() - start_time),
                        episode_rewards_sum/args.testing_episodes_num, episode_length_sum/args.testing_episodes_num))
                    logger.write_results_log(console_f,
                                             time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                                             counter_value,
                                             counter_value / (time.time() - start_time),
                                             episode_rewards_sum / args.testing_episodes_num,
                                             episode_length_sum / args.testing_episodes_num)
                    if args.save_max and (episode_rewards_sum / args.testing_episodes_num) >= max_score:
                        max_score = episode_rewards_sum / args.testing_episodes_num
                        logger.save_policy_model(shared_model, count="max_reward")
                    break

                reward_sum = 0
                episode_length = 0
                actions.clear()
                state = env.reset()

            state = torch.from_numpy(state)
