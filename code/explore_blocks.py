import pddlgym
import imageio
from PIL import Image
import os
from itertools import product
import create_predicates_blocks as cp
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import gc 
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import random
 


TABLE_COLOR    = [128, 51,  0,   255]
ROBOT_COLOR    = [102, 102, 102,  255]

RED_COLOR      = [230, 26,  26,  255] #block a
TURQOISE_COLOR = [112, 227, 246, 255] #block b
BLUE_COLOR     = [41,  28,  167, 255] #block c
GREEN_COLOR    = [38,  221, 41,  255] #block d
YELLOW_COLOR   = [242, 186, 65,  255] #block e
FUCHSIA_COLOR  = [245, 59,  242, 255] #block f

ALL_COLORS = []

#ALL_COLORS.append(TABLE_COLOR)

ALL_COLORS.append(RED_COLOR) #a
ALL_COLORS.append(TURQOISE_COLOR) #b
ALL_COLORS.append(BLUE_COLOR) #c
ALL_COLORS.append(GREEN_COLOR) #d
ALL_COLORS.append(YELLOW_COLOR) #e
ALL_COLORS.append(FUCHSIA_COLOR) #f
ALL_COLORS.append(ROBOT_COLOR)

ALL_COLORS = np.array(ALL_COLORS)

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

        true_predicates = obs.literals
        next_predicates = obs.literals
        while(true_predicates == next_predicates):
            action = env.action_space.sample(obs)
            obs, reward, done, info = env.step(action)
            true_predicates = next_predicates.copy()
            next_predicates = obs.literals
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

def find_unique_quadruples(img):
    unique_quadruples = set()  # Use a set to store unique quadruples
    
    # Iterate through each pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extract the quadruple for the current pixel
            quadruple = tuple(img[i, j, :])
            unique_quadruples.add(quadruple)
    
    # Convert the set of unique quadruples back to a list
    unique_quadruples_list = list(unique_quadruples)
    return unique_quadruples_list

def create_images_for_quadruples(img, unique_quadruples):
    for quadruple in unique_quadruples:
        # Create a white image of the same shape as `img`
        white_img = np.ones_like(img) * 255
        
        # Iterate over each pixel in `img`
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # Check if the current pixel matches the quadruple
                if all(img[i, j, :] == np.array(quadruple)):
                    # Set the pixel in `white_img` to the quadruple's color
                    white_img[i, j, :] = np.array(quadruple)
        
        # Display the image
        plt.figure()
        plt.imshow(white_img)
        plt.title(f"Quadruple: {quadruple}")
        plt.axis('off')
        plt.show()

def create_feature_maps(img, ALL_COLORS):

    feature_maps = np.zeros((480, 480, 7), dtype='b')
    
    for i, color in enumerate(ALL_COLORS):
        # Find pixels exactly matching the current color
        matches = np.all(img == color, axis=-1)
        
        # Update the corresponding feature map
        feature_maps[:, :, i] = matches.astype(np.float32)

    return feature_maps

def plot_feature_map(feature_map, original_img, title="Feature Map"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[1].imshow(feature_map, cmap='gray')
    axs[1].set_title(title)
    for ax in axs:
        ax.axis('off')
    plt.show()


matplotlib.use('Agg')
n_of_resets = 0
env = pddlgym.make("PDDLEnvBlocks-v0", seed=1, dynamic_action_space=True)
obs, debug_info = env.reset()

img = env.render()
env.action_space.seed(0)
feature_maps = create_feature_maps(img, ALL_COLORS)

output_dir = f'dataset_blocks_mnist/train/images/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
imageio.imsave(os.path.join(output_dir, f'image_{(1)}.png'), img)
labels_output_dir = f'dataset_blocks_mnist/train/masks/'
if not os.path.exists(labels_output_dir):
    os.makedirs(labels_output_dir)

np.save(os.path.join(labels_output_dir, f'mask_{1}'), feature_maps)

_ , literals, predicate_labels = cp.extract_grounded_predicates(env)

labels_train = []
labels_train.append(predicate_labels)

plt.close()
#literals, predicate_labels = cp.extract_grounded_predicates(env)

#mnist_quadruples = find_mnist_quadruples(img)

#create_images_for_quadruples(img, mnist_quadruples)
steps = []
step = 0
seen_predicates = []
seen_predicates.append(obs.literals)
seen_actions = {}
seen_actions[obs.literals] = []

for k in range(1, 5500):


    while(obs.literals in seen_predicates):
        action = env.action_space.sample(obs)
        obs, reward, done, info = env.step(action)
        step += 1
        #if step >= 10000 and n_of_resets == 0 :
        #    n_of_resets = 1
        #    continue
        #true_predicates = next_predicates.copy()
        #next_predicates = obs.literals
    '''if step >= 10000 and n_of_resets == 0:
        print("Too many actions. Resetting environment")
        step = 0
        env = pddlgym.make("PDDLEnvBlocks-v0", seed=7)
        #env.seed(0)
        #env.reset()
        #env.action_space.seed(n_of_resets)
        continue'''
    seen_predicates.append(obs.literals)
    img = env.render()
    steps.append(step)
    print(f"Step {k}: {step}")
    step = 0

    feature_maps = create_feature_maps(img, ALL_COLORS)

    output_dir = f'dataset_blocks_mnist/train/images/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    imageio.imsave(os.path.join(output_dir, f'image_{(k+1)}.png'), img)
    labels_output_dir = f'dataset_blocks_mnist/train/masks/'
    if not os.path.exists(labels_output_dir):
        os.makedirs(labels_output_dir)
    np.save(os.path.join(labels_output_dir, f'mask_{k+1}'), feature_maps)
            
    _, literals, predicate_labels = cp.extract_grounded_predicates(env)
    labels_train.append(predicate_labels)

    plt.close('all')
 
labels_train_df = pd.DataFrame(labels_train, columns=literals)
labels_train_df.to_csv(f'dataset_blocks_mnist/train/labels.csv', index=False, sep=';')
with open('steps.txt', 'w') as file:
    file.write('\n'.join(str(s) for s in steps))



#Create Test Set

labels_test = []

for k in range(1500):

    while(obs.literals in seen_predicates):
        action = env.action_space.sample(obs)
        obs, reward, done, info = env.step(action)
    seen_predicates.append(obs.literals)
    img = env.render()

    feature_maps = create_feature_maps(img, ALL_COLORS)

    output_dir = f'dataset_blocks_mnist/test/images/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    imageio.imsave(os.path.join(output_dir, f'image_{(k+1)}.png'), img)
    labels_output_dir = f'dataset_blocks_mnist/test/masks/'
    if not os.path.exists(labels_output_dir):
        os.makedirs(labels_output_dir)
    np.save(os.path.join(labels_output_dir, f'mask_{k+1}'), feature_maps)

    _, literals, predicate_labels = cp.extract_grounded_predicates(env)
    labels_test.append(predicate_labels)

    plt.close('all')

labels_test_df = pd.DataFrame(labels_test, columns=literals)
labels_test_df.to_csv('dataset_blocks_mnist/test/labels.csv', index=False, sep=';')