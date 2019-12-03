from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import random
from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import chain, cycle, islice




class MyIterableDataset(IterableDataset):
    def __init__(self, data_list, batch_size=4):
        self.data_list = data_list
        self.batch_size = batch_size 
        
    @property
    def shuffled_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    @classmethod
    def split_datasets(cls, data_list, batch_size, max_workers):
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break 
        
        split_size = batch_size // num_workers
        return [cls(data_list, batch_size=split_size) for _ in range(num_workers)]

    def process_data(self, data):
        for x in data:
            yield x
    
    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list) for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data_list) // worker_info.num_workers
    dataset.data_list = dataset.data_list[worker_id * split_size: (worker_id+1) * split_size]


class MultiStreamDataLoader:
    def __init__(self, datasets):
        self.datasets = datasets
    
    def get_stream_loaders(self):
        return zip(*[
            DataLoader(dataset, num_workers=1, batch_size=None)
            for dataset in self.datasets 
        ])
    
    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield list(chain(*batch_parts))



if __name__ == '__main__':

    #Example 1: simple lists
    """
    data_list = [
        range(i * 10, i * 10 + 7) for i in range(30)
    ]
    """

    #Example 2: gym environments
    import gym
    import numpy as np
    import cv2
 

    class GymEnv(object):
        def __init__(self, env_name, niter=1000):
            self.env = gym.make(env_name)
            self.niter = niter
            self.action = np.random.randint(0, self.env.action_space.n)
        
        def __iter__(self):
            self.env.reset()
            for _ in range(self.niter):
                action = np.random.randint(0, self.env.action_space.n)
                observation, reward, done, info = self.env.step(action)
                if done: 
                    self.env.reset()
                    yield self.env.step(action)
                else:
                    yield observation, reward, done, info
                

    envs = ['SpaceInvaders-v0', 'Pong-v0', 'Breakout-v0', 'Alien-v0', 'Asterix-v0', 'Amidar-v0']
    #envs = ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk', 'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kaboom', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan', 'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing', 'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']

    data_list = []
    for i in range(20):
        env_name = envs[i%len(envs)]
        data_list.append(GymEnv(env_name))
    random.shuffle(data_list)

    # iterable_dataset = MyIterableDataset(data_list, batch_size=4)
    # loader = DataLoader(iterable_dataset, batch_size=None, num_workers=2, worker_init_fn=worker_init_fn)

    datasets = MyIterableDataset.split_datasets(data_list, batch_size=5, max_workers=2)
    loader = MultiStreamDataLoader(datasets)

    for batch in loader:
        # Example 1
        # print(batch)
        for j in range(4):
            obs, reward, done, info = batch[j]
            obs = obs.numpy()
            cv2.imshow('env'+str(j), obs.astype(np.uint8)[..., ::-1])
        cv2.waitKey(5)
    
