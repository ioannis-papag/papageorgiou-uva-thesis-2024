import pddlgym
import imageio
from PIL import Image
import os
from pddlgym_planners.ff import FF
from itertools import product
import create_predicates as cp
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np




'''(define (domain sokoban)
  (:requirements :typing )
  (:types thing location direction)
  (:predicates (move-dir ?v0 - location ?v1 - location ?v2 - direction)
	(is-nongoal ?v0 - location)
	(clear ?v0 - location)
	(is-stone ?v0 - thing)
	(at ?v0 - thing ?v1 - location)
	(is-player ?v0 - thing)
	(at-goal ?v0 - thing)
	(move ?v0 - direction)
	(is-goal ?v0 - location)
  )'''


def pretty_print_frozenset(fs):
    sorted_list = sorted(fs)
    pretty_string = "\n".join(str(item) for item in sorted_list)
    print(pretty_string)

def random_walk(env_name, num_steps=100):
    start_time = time.time()
    env = pddlgym.make(env_name)
    observation , debug_info = env.reset()
    img = env.render()
    output_dir = f'dataset'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    imageio.imsave(os.path.join(output_dir, f'image_0.png'), img)
    literals, predicate_labels = cp.extract_grounded_predicates(env)

    labels = []
    labels.append(predicate_labels)

    for step in range(num_steps):
        #print(f"Step {step+1}")

        action = env.action_space.sample(observation)

        observation, reward, done, info = env.step(action)

        img = env.render()
        output_dir = f'dataset'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imageio.imsave(os.path.join(output_dir, f'image_{step+1}.png'), img)

        literals, predicate_labels = cp.extract_grounded_predicates(env)
        labels.append(predicate_labels)

        #print(f"Step {step}: Action: {action}, Reward: {reward}")

        if done:
            print("Reached a terminal state. Resetting environment.")
            env.reset()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time for 10 steps: {elapsed_time} seconds")

    env.close()
    start_time = time.time()
    labels_df = pd.DataFrame(labels, columns=literals)
    labels_df.to_csv('dataset/labels.csv', index=False, sep=';')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time for writing the CSV: {elapsed_time} seconds")





#env_name = "PDDLEnvSokoban-v0"
#random_walk(env_name)

env = pddlgym.make("PDDLEnvSokoban-v0")
obs, debug_info = env.reset()
img = env.render()
#literals, predicate_labels = cp.extract_grounded_predicates(env)


imageio.imsave("frame1.png", img)
block_height = img.shape[0] // 9
block_width = img.shape[1]// 10
done = False
k = 0


for k in range(30):

    action = env.action_space.sample(obs)
    obs, reward, done, debug_info = env.step(action)
    img = env.render()


    for i in range(9):
        for j in range(10):
            start_row = i * block_height
            end_row = (i + 1) * block_height
            start_col = j * block_width
            end_col = (j + 1) * block_width
            block = img[start_row:end_row, start_col:end_col]

            literals = obs.literals

            if j + 1 < 10:
                coords = '0' + str(j+1)  + '-0' + str(i+1)
            else:
                coords = '10' + '-0' + str(i+1)

            literal_string = ''
            for item in literals:
                if 'move-dir' in str(item):
                    continue
                if coords in str(item):
                    literal_string += str(item)

            
            label = ''
            if 'player' in literal_string:
                label = "player"
            elif 'stone' in literal_string:
                label = "stone"
            elif 'is-goal' in literal_string:
                label = "goal"
            elif 'clear' in literal_string:
                label = "path"
            else:
                label = "wall"

            output_dir = f'cnn_dataset/test/{label}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            imageio.imsave(os.path.join(output_dir, f'block_{(k+1) * (10*i + j)}.png'), block)

            

            output_dir = f'images/image_{k}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            imageio.imsave(os.path.join(output_dir, f'block_{i}_{j}.png'), block)

    #imageio.imsave(os.path.join(output_dir, f'full_image_{k}.png'), img)'''
#412 x 450 x 4'''