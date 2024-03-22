import pddlgym
import imageio
from PIL import Image
import os
from pddlgym_planners.ff import FF
from itertools import product
import create_predicates_blocks as cp
import pandas as pd
import time
import matplotlib.pyplot as plt




'''(define (domain blocks)
    (:requirements :strips :typing)
    (:types block robot)
    (:predicates 
        (on ?x - block ?y - block)
        (ontable ?x - block)
        (clear ?x - block)
        (handempty ?x - robot)
        (handfull ?x - robot)
        (holding ?x - block)
        (pickup ?x - block)
        (putdown ?x - block)
        (stack ?x - block ?y - block)
        (unstack ?x - block)
    )'''


def pretty_print_frozenset(fs):
    sorted_list = sorted(fs)
    pretty_string = "\n".join(str(item) for item in sorted_list)
    print(pretty_string)

def random_walk(env_name, num_steps=10):
    start_time = time.time()
    env = pddlgym.make(env_name, seed=2024)
    env.seed(2024)
    observation , debug_info = env.reset()
    img = env.render()
    output_dir = f'dataset_blocks_full'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    imageio.imsave(os.path.join(output_dir, f'image_0.png'), img)
    objectos, literals, predicate_labels = cp.extract_grounded_predicates(env)

    labels = []
    labels.append(predicate_labels)

    for step in range(num_steps):
        #print(f"Step {step+1}")

        action = env.action_space.sample(observation)

        observation, reward, done, info = env.step(action)

        img = env.render()
        output_dir = f'dataset_blocks_full'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imageio.imsave(os.path.join(output_dir, f'image_{step+1}.png'), img)

        objectos, literals, predicate_labels = cp.extract_grounded_predicates(env)
        labels.append(predicate_labels)

        #print(f"Step {step}: Action: {action}, Reward: {reward}")
        plt.close()
        if done:
            print("Reached a terminal state. Resetting environment.")
            env.reset()
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time for 10 steps: {elapsed_time} seconds")

    env.close()
    start_time = time.time()
    labels_df = pd.DataFrame(labels, columns=literals)
    labels_df.to_csv('dataset_blocks_full/labels.csv', index=False, sep=';')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time for writing the CSV: {elapsed_time} seconds")


env_name = "PDDLEnvBlocks-v0"
random_walk(env_name)