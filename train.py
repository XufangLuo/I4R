import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from collections import deque

from envs import create_atari_env
from model import ActorCritic
from regularization import MaxDivideMin
from regularization import MaxMinusMin


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, logger, optimizer=None):
    if args.save_sigmas:
        sigmas_f = logger.init_one_sigmas_file(rank)

    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    if args.add_rank_reg:
        if args.rank_reg_type == "maxdividemin":
            rank_reg = MaxDivideMin.apply
        elif args.rank_reg_type == "maxminusmin":
            rank_reg = MaxMinusMin.apply

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    local_counter = 0
    episode_length = 0
    while True:
        if args.max_counter_num != 0 and counter.value > args.max_counter_num:
            exit(0)
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        if args.add_rank_reg:
            hiddens = [None] * 2  # 0: last layer, 1: last last layer

        for step in range(args.num_steps):
            episode_length += 1
            model_inputs = (Variable(state.unsqueeze(0)), (hx, cx))
            if args.add_rank_reg:
                value, logit, (hx, cx), internal_features = model(model_inputs, return_features=True)
            else:
                value, logit, (hx, cx) = model(model_inputs)

            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            if args.add_rank_reg:
                if hiddens[0] is None:
                    hiddens[0] = internal_features[-1]
                    hiddens[1] = internal_features[-2]
                else:
                    hiddens[0] = torch.cat([hiddens[0], internal_features[-1]])
                    hiddens[1] = torch.cat([hiddens[1], internal_features[-2]])

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            local_counter += 1
            with lock:
                if local_counter % 20 == 0:
                    counter.value += 20

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = args.gamma * values[i + 1].data - values[i].data + rewards[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        total_loss = policy_loss + args.value_loss_coef * value_loss

        # internal layers regularizer
        retain_graph = None
        if args.add_rank_reg:
            current_rankreg_coef = args.rank_reg_coef
            # total_loss = total_loss + rank_reg(hiddens[0], args.rank_reg_coef)
            if args.save_sigmas and local_counter % args.save_sigmas_every <= 3:
                norm = rank_reg(hiddens[0], current_rankreg_coef, counter.value, sigmas_f, logger)
            else:
                norm = rank_reg(hiddens[0], current_rankreg_coef)
            total_loss = total_loss + norm

        optimizer.zero_grad()

        total_loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
