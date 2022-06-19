#Import necessary packages
from unityagents import UnityEnvironment
import numpy as np
import random
import argparse
import torch
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent_cc import Agent
import time, os, fnmatch, shutil

import warnings
warnings.filterwarnings('ignore')

"""Training code -> train()
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        num_agents (int): number of parallel environments can be 1 or 20
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    
"""
def train(n_episodes=2000, max_t=1000, num_agents=20, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    # Select the Unity environment based on the number of agents.
    # Change the file_name parameter to match the location of the Unity environment that you downloaded.
    # Keeping 'no_graphics=True' helps reduce training time. Remove it to be able to visualise the environment while training.
    if(num_agents == 1):
        env = UnityEnvironment(file_name="Reacher_Windows_x86_64_1/Reacher.exe", no_graphics=True, seed=0)
    else:
        env = UnityEnvironment(file_name="Reacher_Windows_x86_64_20/Reacher.exe", no_graphics=True, seed=0)
    
    print("\n")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment for train
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    # Initialize an agent
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)

    scores_all = []                       # list containing scores from each episode
    scores_window = deque(maxlen=100)     # last 100 scores
    eps = eps_start                       # initialize epsilon
    max_mean = -np.inf
    i_episode_solved = 0
    first_solve = True
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations # get the current state (for each agent)
        agent.reset()                         # reset Noise
        scores_episode = np.zeros(num_agents) # initialize the score (for each agent)
        for t in range(max_t):
            actions = agent.act(states, eps, add_noise=True)  # select an action (for each agent)
            env_info = env.step(actions)[brain_name]     # send all actions to tne environment
            next_states = env_info.vector_observations   # get next state (for each agent)
            rewards = env_info.rewards                   # get reward (for each agent)
            dones = env_info.local_done                  # see if episode has finished (for each agent)
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, num_agents) 
            states = next_states                         # roll over states to next time step
            scores_episode += rewards                    # update the score (for each agent)
            if np.any(dones):                            # exit loop if episode finished
                break 
        scores_window.append(np.mean(scores_episode))
        scores_all.append(np.mean(scores_episode))
        eps = max(eps_end, eps_decay*eps)                # decrease epsilon for off-policy epsilon greediness
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
        #Store the weights corresponding to the best mean score
        curr_mean = np.mean(scores_window)
        if ((curr_mean >= 30.0) and (curr_mean > max_mean)):
            max_mean = curr_mean
            if (first_solve == True):
                first_solve = False
                i_episode_solved = i_episode
            #Save the best checkpoints
            torch.save(agent.actor_local.state_dict(), 'checkpoints/checkpoint_actor_cc_'+ str(num_agents) + '_agent.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoints/checkpoint_critic_cc_'+ str(num_agents) + '_agent.pth')
            if ((n_episodes % 500 == 0) and (curr_mean >= 30.0)):
                break
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode_solved - 100, max_mean))
    env.close()
    return scores_all

"""Testing code -> test()
    
    Params
    ======
        actor_checkpoint_path: File path for the saved model checkpoints of actor network
        critic_checkpoint_path: File path for the saved model checkpoints of critic network
        num_agents (int): number of parallel environments can be 1 or 20
    
"""
def test(actor_checkpoint_path, critic_checkpoint_path, num_agents=20):
    # Select the Unity environment based on the number of agents.
    # Change the file_name parameter to match the location of the Unity environment that you downloaded.
    # Keeping 'no_graphics=True' helps reduce training time. Remove it to be able to visualise the environment while training.  
    if(num_agents == 1):
        env = UnityEnvironment(file_name="Reacher_Windows_x86_64_1/Reacher.exe")
    else:
        env = UnityEnvironment(file_name="Reacher_Windows_x86_64_20/Reacher.exe")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment for test
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    # Initialize an agent
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
    
    # load the weights from file
    agent.actor_local.load_state_dict(torch.load(actor_checkpoint_path))
    agent.critic_local.load_state_dict(torch.load(critic_checkpoint_path))
    
    #Run the evaluation
    states = env_info.vector_observations # get the current state (for each agent)
    scores_episode = np.zeros(num_agents) # initialize the score (for each agent)

    for t in range(1000):
        actions = agent.act(states, eps=1, add_noise=False)  # select an action (for each agent)
        env_info = env.step(actions)[brain_name]             # send all actions to tne environment
        next_states = env_info.vector_observations           # get next state (for each agent)
        rewards = env_info.rewards                           # get reward (for each agent)
        dones = env_info.local_done                          # see if episode has finished (for each agent)
        states = next_states                                 # roll over states to next time step
        scores_episode += rewards                            # update the score (for each agent)
        if np.any(dones):                                    # exit loop if episode finished
            break 

    print('\nTotal score (averaged over {} agents) in Test: {}'.format(num_agents, np.mean(scores_episode)))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch training script')
    
    parser.add_argument('--evaluate_actor', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate for actor (file name)')
    parser.add_argument('--evaluate_critic', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate for critic (file name)')
    parser.add_argument('--n_episodes', default=2000, type=int, metavar='N', help='maximum number of training episodes')
    parser.add_argument('--max_t', default=1000, type=int, metavar='N', help='maximum number of timesteps per episode')
    parser.add_argument('--num_agents', default=20, type=int, metavar='N', help='number of parallel environments can be 1 or 20')
    parser.add_argument('--eps_start', default=1.0, type=float, metavar='N', help='starting value of epsilon')
    parser.add_argument('--eps_end', default=0.01, type=float, metavar='N', help='minimum value of epsilon')
    parser.add_argument('--eps_decay', default=0.995, type=float, metavar='N', help='multiplicative factor for epsilon decay')
    
    args = parser.parse_args()
    
    if((args.num_agents != 1) and (args.num_agents != 20)):
        print("\nnum_agents can be either 1 or 20\n")
        exit(0)

    if ((args.evaluate_actor) and (args.evaluate_critic)):
        print("\nRunning Test with the below parameters:\n")
        print("checkpoint path for actor = ", args.evaluate_actor)
        print("checkpoint path for critic = ", args.evaluate_critic)
        print("num_agents = ", args.num_agents)
        print("\n")
        test(args.evaluate_actor, args.evaluate_critic, args.num_agents)
        exit(0)
    else:
        print("\nRunning Train with the below parameters:\n")
        print("n_episodes = ", args.n_episodes)
        print("max_t = ", args.max_t)
        print("num_agents = ", args.num_agents)
        print("eps_start = ", args.eps_start)
        print("eps_end = ", args.eps_end)
        print("eps_decay = ", args.eps_decay)
        print("\n")
        scores = train(args.n_episodes, args.max_t, args.num_agents, args.eps_start, args.eps_end, args.eps_decay)
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        fig.savefig("plots/training_plot_" + str(args.num_agents) + ".pdf", dpi=fig.dpi)
        exit(0)