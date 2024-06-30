import pddlgym
import imageio
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import h5py
import argparse

def extract_all_predicates(objects, literals):
    """
    Creates a list of all the predicate names and
    truth values and returns them

    Parameters:
    - objects: list of size 7 containing the names of all objects in the state
    - literals: frozen set produced by the PDDLGym API containing all true predicates
                in the current state
    Output:
    - all_predicates: list of size 70 containing the names of all possible
                    predicate-argument combinations
    - predicate_labes: list of size 70 containing the corresponding truth
                    value of each of the predicates in all_predicates
    """

    #Sort the objects' names to ensure order
    objects = sorted(objects)
    all_predicates = []

    #Create the names of all unary predicates
    for object_1 in objects:
        all_predicates.append('holding(' + object_1 + ')')
        all_predicates.append('clear(' + object_1 + ')')
        all_predicates.append('ontable(' + object_1 + ')')

    #Create the name of all binary predicate combinations
    for object_1 in objects:
        for object_2 in objects:
            all_predicates.append('on(' + object_1 + ',' + object_2 + ')')

    #Initialize the values of predicates
    predicate_labels = [0 for item in all_predicates]

    #Create a list of all the true predicates in the state
    literals_list = list(literals)
    literals_string_list = []
    for lit in literals_list:
        literals_string_list.append(str(lit))

    #If the predicate is in the above list, set its value to True in the list to be returned
    for i, predicate in enumerate(all_predicates):
        if predicate in literals_string_list:
            predicate_labels[i] = 1


    return all_predicates, predicate_labels

def extract_grounded_predicates(env):
    """
    Returns all predicates (True and False) from
    a certain state of the environment along with 
    their labels and corresponding objects

    Parameters:
    - env: PDDLGym environment at its current state

    Output:
    - objects: list of size 7 containing all the objects of the state
    - lits: list of size 70 containing binary values corresponding to 
            the truth value of each predicate 
    - predicate_labels: list of size 70 of strings containing the name
                        of each predicate in lits
    """
    
    #Get all the objects of the environemnt
    state = env.get_state()
    objects = list(state.objects)

    lits, predicate_labels = extract_all_predicates(objects, state.literals)

    return objects, lits, predicate_labels

def process_image_and_objects(image, coords_dictt):
    """
    Process the input image and object coordinates to produce an extended 
    image with binary masks and an array with the coordinates.

    Input:
    - image: np.array of shape (4, 480, 480), the input image with 4 color
            channels.
    - coords_dict: dict of shape (N, 4), with the coordinates of N objects
                    in the format {object_id: [x1, y1, x2, y2]}.

    Output:
    - extended_image: torch.Tensor of shape (4 + N, 480, 480), the extended
                    image with binary masks.
    - coords_array: torch.Tensor of shape (1 + N, 4), the array with coordinates.
    """
    
    #Convert numpy image to torch tensor
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    coords_dict = dict(sorted(coords_dictt.items()))
     
    #Number of objects
    N = 7
    
    #Initialize the extended image tensor with original image channels
    extended_image = torch.zeros((4 + N, 480, 480), dtype=image_tensor.dtype)
    extended_image[:4] = image_tensor
    
    #Initialize the coordinates tensor
    coords_array = torch.zeros((1 + N, 5), dtype=torch.float32)


    #Initialize min and max coordinates to find the smallest bounding box containing all others
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    #Iterate through the coordinates to find the minimum and maximum values
    for object_id, (x1, y1, x2, y2) in coords_dict.items():
        min_x = min(min_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)

    height, width, channels = 480, 480, 4

    #Calculate the smallest box coordinates
    coords_array[0] = torch.tensor([0, min_x* width / 3.2, min_y* height / 3.2, max_x* width / 3.2, max_y* height / 3.2], dtype=torch.float32)


    #Process each object to create binary masks
    for i, (object_id, (x1, y1, x2, y2)) in enumerate(coords_dict.items(), start=1):

        #Normalize coordinates to pixels based on PDDLGym values
        x1 = int(x1 * width / 3.2)  
        y1 = int(y1 * height / 3.2)  
        x2 = int(x2 * width / 3.2)  
        y2 = int(y2 * height / 3.2) 

        mask = torch.zeros((480, 480), dtype=torch.float32)

        #Create the binary mask for the object
        mask[y1:y2, x1:x2] = 1.0

        #Flip needed due to how the axis are defined in PDDLGym
        mask = torch.flip(mask, dims=[0])

        #Save the masks and coordinates
        extended_image[4 + i - 1] = mask
        coords_array[i] = torch.tensor([i, x1, y1, x2, y2], dtype=torch.float32)


    return extended_image, coords_array


def main(args):

    init_seed = args.seed
    num_steps = args.length
    isLayoutUnique = args.unique_layout

    if isLayoutUnique:
        num_steps = 7000


    matplotlib.use('Agg') #Important to not run out of memory from rendering images
    
    #Create environment and set seeds
    rng = np.random.default_rng(seed=init_seed)
    env = pddlgym.make("PDDLEnvBlocks-v0", seed=init_seed, dynamic_action_space=True)
    obs, debug_info = env.reset()
    env.action_space.seed(init_seed-1)
    np.random.seed(init_seed)


    #Create output directories
    output_dir = f'dataset_blocks/images/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    output_dir_img = f'dataset_blocks/imgs'
    if not os.path.exists(output_dir_img):
        os.makedirs(output_dir_img)
    file_path_img = os.path.join(output_dir_img, 'imgs.h5')


    output_dir_box = f'dataset_blocks/boxes'
    if not os.path.exists(output_dir_box):
        os.makedirs(output_dir_box)
    file_path_box = os.path.join(output_dir_box, 'boxes.h5')


    seen_predicates = [] #Used to track all the visited states
    labels = []
        
    with h5py.File(file_path_img, 'w') as img_hf, h5py.File(file_path_box, 'w') as box_hf:
        for k in tqdm(range(num_steps)):
            if k > 0:
                #If it is not the first step, take an action and move to a new state
                #While the state has been already visited, take a new action
                if isLayoutUnique:
                    while obs in seen_predicates:
                        action = env.action_space.sample(obs)
                        obs, reward, done, info = env.step(action)
                else:
                    action = env.action_space.sample(obs)
                    obs, reward, done, info = env.step(action)           

            img, clothes, obj_coords = env.render(rng_gen=rng, mnist=False) #Render the image

            if isLayoutUnique:
                #If every image should be unique, keep track of all the seen states
                seen_predicates.append(obs.literals)

            #Extract the bounding boxes and binary masks of objects    
            extended_image, coords_array = process_image_and_objects(img, obj_coords)

            #Convert them to numpy arrays
            extended_image_np = extended_image.numpy() if torch.is_tensor(extended_image) else extended_image
            coords_array_np = coords_array.numpy() if torch.is_tensor(coords_array) else coords_array

            #Save the binary masks and bounding boxes
            img_hf.create_dataset(f'image_torch_{k+1}', data=extended_image_np, compression='gzip')
            box_hf.create_dataset(f'box_torch_{k+1}', data=coords_array_np, compression='gzip')
            
            #Save the image
            imageio.imsave(os.path.join(output_dir, f'image_{(k+1)}.png'), img)
            
            #Extract all the predicates from the image in the form of a vector of length 70
            _, literals, predicate_labels = extract_grounded_predicates(env)

            labels.append(predicate_labels)

            #Close all figures to save memory
            plt.close('all')

        #Save all the labels/targets as a csv file
        labels_df = pd.DataFrame(labels, columns=literals)
        labels_df.to_csv(f'dataset_blocks/labels.csv', index=False, sep=';')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("-s" , "--seed", type=int, help="Sets the random seed for all operations", default=1)
    parser.add_argument("-n", "--length", type=int, help="Length of the dataset to be generated", required=False, default=7000)
    parser.add_argument("-l", "--unique_layout", type=bool, help="If True, all the samples created contain unique layouts", default="True")

    args = parser.parse_args()
    main(args)

