# Copyright 2020, Prof. Marko Orescanin, NPS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  
# Created by marko.orescanin@nps.edu on 7/21/20
#
# Modified by Brian Atkinson on 1/5/21
# This module contains all parameter handling for the project, including
# all command line tunable parameter definitions, any preprocessing
# of those parameters, or generation/specification of other parameters.


import argparse
import os
import yaml
import time


def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run training for FlappyBird reinforcement learning with human influence.')
    
    #render the screen
    parser.add_argument('--render', type=str2bool, default=False,
                        help='if True, the game will be displayed')    
    #network arguments
    
    
    #hyperparameters
    parser.add_argument('--batch_size', type=int, default=30,
                        help="number of episodes to conduct rmsprop parameter updates over")
    parser.add_argument('--dropout', type=float, default=0,
                        help="likelihood of a neuron to be dropped, as a decimal from [0,1)")
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--hidden', type=int, default=200,
                        help="the number of hidden nodes to use in the network")
    parser.add_argument('--leaky', type=str2bool, default=False,
                        help="if True, use Leaky ReLu activation, else use ReLU")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="specify the base learning rate for the model")
    parser.add_argument('--L2', type=float, default=1e-3,
                        help='amount to decay weights for L2 penalty in Adam')
    parser.add_argument('--num_hiddens', type=int, default=2,
                        help="number of hidden linear layers to use")     
    parser.add_argument('--optim', type=str, default='Adam',
                        help='String specifying the type of optimizer to be used.')
    parser.add_argument('--sigmoid', type=str2bool, default=False,
                        help="if True, uses sigmoid activation for output layer of network")
    parser.add_argument('--softmax', type=str2bool, default=False,
                        help="if True, uses softmax activation for output layer of network")
    parser.add_argument("--temperature", type=float, default=1,
                        help="the temperature value used to modify input to the sigmoid activation \
                            function. must be greater than 0")
    
    
    #training arguments
    parser.add_argument('--maximize', type=str2bool, default=True,\
                        help="if True, Adam will maximize the objective function. Not implemented for RMSprop")
    parser.add_argument('--model_type', type=str, default='PGNetwork',
                        help="can be PGNetwork or NoisyPG")
    parser.add_argument('--num_episodes', type=int, default=15000,
                        help="the number of episodes to train the agent on")
    parser.add_argument('--save_stats', type=int, default=200,
                        help="specifies the number of episodes to wait until saving network parameters, training summaries, and moves")
    parser.add_argument('--seed', type=int, default=1,
                        help="specify a number to seed the PRNGs with")
    
    
    #filepath arguments
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(),'data/debug/'),
                        help='a filepath to an existing directory to save to')
    
    # multi-gpu training arguments
    parser.add_argument('--mgpu_run', type=str2bool, default=False,
                        help="multi gpu run")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="number of gpu's on the machine, juno is 2")
    
    # multi-processing arguments
    parser.add_argument('--use_multiprocessing', type=str2bool, default=True,
                        help="specifys weather to use use_multiprocessing in .fit_genrator method ")
    parser.add_argument("--workers", type=int, default=6,
                        help="number of CPU's, for my machine 6 workers, for Juno 18")
    
    return parser.parse_args()


# you have to use str2bool
# because of an issue with argparser and bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_hparams():
    """any preprocessing, special handling of the hparams object"""

    parser = make_argparser()

    return parser


def save_hparams(hparams):
    path_ = os.path.join(hparams.model_dir, 'params.txt')
    hparams_ = vars(hparams)
    with open(path_, 'w') as f:
        for arg in hparams_:
            print(arg, ':', hparams_[arg])
            f.write(arg + ':' + str(hparams_[arg]) + '\n')

    path_ = os.path.join(hparams.model_dir, 'params.yml')
    with open(path_, 'w') as f:
        yaml.dump(hparams_, f,
                  default_flow_style=False)  # save hparams as a yaml, since that's easier to read and use