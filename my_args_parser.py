import argparse

# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=16,
                    help='how many training processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=100000,
                    help='maximum length of an episode (default: 100000)')
parser.add_argument('--env-name', default='atari-PongDeterministic-v4',
                    help='environment to train on (default: atari-PongDeterministic-v4)')
parser.add_argument('--no-shared', type=int, default=0,
                    help='use an optimizer without shared momentum. (default: 0) (0:False; 1:True)')
# some added seetings
parser.add_argument('--log-dir', default="debug",
                    help='log folder name. (debug)')
parser.add_argument('--log-dir-pre', default="./logs/",
                    help='log folder prefix. (./logs/)')
parser.add_argument('--max-counter-num', type=int, default=50100000,
                    help='max counter number to end. (default: 50100000)')
parser.add_argument('--save-policy-models', type=int, default=1,
                    help='save policy models or not. (default: 1) (0:False; 1:True)')
parser.add_argument('--save-policy-models-every', type=int, default=1000000,
                    help='save policy models every X steps. (default: 1000000)')
parser.add_argument('--save-max', default=True,
                    metavar='SM', help='Save model on every test run high score matched or bested')

# Testing settings
parser.add_argument('--testing-episodes-num', type=int, default=3,
                    help='the number of testing episodes. (3)')
parser.add_argument('--testing-every-counter', type=int, default=50000,
                    help='for test by counter: test every [this value]. (50000)')

# optimize representation settings
parser.add_argument('--add-rank-reg', type=int, default=0,
                    help='add regularization of last hidden layer to encourage expressive representations. '
                         '(default: 0, not add)')
parser.add_argument('--rank-reg-type', default="maxminusmin",
                    help='maxdividemin, maxminusmin. (maxminusmin)')
parser.add_argument('--rank-reg-coef', type=float, default=0.001,
                    help='rank related regularization coefficient (default: 0.001)')
parser.add_argument('--save-sigmas', type=int, default=0,
                    help='save sigmas for each process during training. (default: 0, not add)')
parser.add_argument('--save-sigmas-every', type=int, default=1500,  # atari: 1500, default: 300
                    help='save sigmas every local_counter. (default: 300)')

def get_args():
    return parser.parse_args()
