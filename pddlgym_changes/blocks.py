from .utils import fig2data

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from tensorflow.keras.datasets import fashion_mnist

def initialize_fashion_mnist():
    # Load the dataset
    (imgs, labels), (_, _) = fashion_mnist.load_data()

    # Dictionary to store available indices for each class
    class_indices = {i: np.where(labels == i)[0].tolist() for i in range(10)}

    return imgs, class_indices

_mnist_images, _class_indices_mnist = initialize_fashion_mnist()

def get_objects_from_obs(obs, rng_gen):
    on_links = {}
    pile_bottoms = set()
    all_objs = set()
    holding = None
    for lit in obs:
        if lit.predicate.name.lower() == "ontable":
            pile_bottoms.add(lit.variables[0])
            all_objs.add(lit.variables[0])
        elif lit.predicate.name.lower() == "on":
            on_links[lit.variables[1]] = lit.variables[0]
            all_objs.update(lit.variables)
        elif lit.predicate.name.lower() == "holding":
            holding = lit.variables[0]
            all_objs.add(lit.variables[0])
    all_objs = sorted(all_objs)

    bottom_to_pile = {}
    for obj in pile_bottoms:
        bottom_to_pile[obj] = [obj]
        key = obj
        while key in on_links:
            assert on_links[key] not in bottom_to_pile[obj]
            bottom_to_pile[obj].append(on_links[key])
            key = on_links[key]

    piles = []
    #Assign a unique color and image to each object
    for item in all_objs:
        block_name_to_color(item)
        block_name_to_image(item, rng_gen)

    for pile_base in all_objs:
        if pile_base in bottom_to_pile:
            piles.append(bottom_to_pile[pile_base])
        else:
            piles.append([])

    return piles, holding

_block_coordinates = {}

def get_block_params(piles, width, height, table_height, robot_height):
    num_blocks = len(piles)
    horizontal_padding = 0.025 * width
    block_width = width / num_blocks - 2*horizontal_padding
    block_height = (height - table_height - robot_height) / num_blocks - 0.05 * height

    block_positions = {}
    for pile_i, pile in enumerate(piles):
        x = horizontal_padding + pile_i * (block_width + 2*horizontal_padding)
        for block_i, name in enumerate(pile):
            y = table_height + block_i * block_height
            block_positions[name] = (x, y)

            _block_coordinates[name] = [x, y, x+block_width, y+block_height]

    return block_width, block_height, block_positions

def draw_table(ax, width, table_height):
    rect = patches.Rectangle((0,0), width, table_height, 
        linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=(0.5,0.2,0.0))
    ax.add_patch(rect)

def draw_robot_mnist(ax, robot_width, robot_height, midx, midy, holding, block_width, block_height, rng_gen):
    x = midx - robot_width/2
    y = midy - robot_height/2
    rect_1 = patches.Rectangle((x,y), robot_width, robot_height, 
        linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=(0.4, 0.4, 0.4))
    ax.add_patch(rect_1)
    rect_1.set_zorder(1)

    _block_coordinates["robot:robot"] = [x, y, x+robot_width, y+robot_height]

    # Holding
    if holding is None:
        holding_color = (1., 1., 1.)
        ec = (0., 0., 0., 0.)
    else:
        #holding_color = block_name_to_color(holding)
        holding_image = _mnist_images[block_name_to_image(holding, rng_gen)]
        ec = (0.2,0.2,0.2)
    holding_x = midx - block_width/2
    holding_y = y - block_height/3
    #rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
    #    linewidth=1, edgecolor=ec, facecolor=holding_color)
    if holding is None:
        rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
        linewidth=1, edgecolor=ec, facecolor=holding_color)
        ax.add_patch(rect)
    else:

        image_artist = ax.imshow(holding_image, extent=[holding_x, holding_x + block_width, holding_y, holding_y + block_height], cmap='gray')
        image_artist.set_zorder(2)
        rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
            linewidth=1, edgecolor=ec, facecolor='none')
        ax.add_patch(rect)
        rect.set_zorder(1)
        _block_coordinates[holding] = [holding_x, holding_y, holding_x + block_width, holding_y + block_height]
    #ax.add_patch(rect)

def draw_robot(ax, robot_width, robot_height, midx, midy, holding, block_width, block_height):
    x = midx - robot_width/2
    y = midy - robot_height/2
    rect_1 = patches.Rectangle((x,y), robot_width, robot_height, 
        linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=(0.4, 0.4, 0.4))
    ax.add_patch(rect_1)
    rect_1.set_zorder(1)

    _block_coordinates["robot:robot"] = [x, y, x+robot_width, y+robot_height]

    # Holding
    if holding is None:
        holding_color = (1., 1., 1.)
        ec = (0., 0., 0., 0.)
    else:
        holding_color = block_name_to_color(holding)
        ec = (0.2,0.2,0.2)
    holding_x = midx - block_width/2
    holding_y = y - block_height/3
    rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
       linewidth=1, edgecolor=ec, facecolor=holding_color)
    if holding is None:
        rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
        linewidth=1, edgecolor=ec, facecolor=holding_color)
        ax.add_patch(rect)
    else:

        #image_artist = ax.imshow(holding_image, extent=[holding_x, holding_x + block_width, holding_y, holding_y + block_height], cmap='gray')
        #image_artist.set_zorder(2)
        #rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
        #    linewidth=1, edgecolor=ec, facecolor='none')
        #ax.add_patch(rect)
        #rect.set_zorder(1)
        rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
        linewidth=1, edgecolor=ec, facecolor=holding_color)
        ax.add_patch(rect)
        _block_coordinates[holding] = [holding_x, holding_y, holding_x + block_width, holding_y + block_height]
    #ax.add_patch(rect)

_block_name_to_color = {}
_block_name_to_image = {}
_block_name_to_class = {}
_rng = np.random.RandomState(0)

def block_name_to_color(block_name):
    if block_name not in _block_name_to_color:
        if len(_block_name_to_color) == 0:
            best_color = (0.9, 0.1, 0.1)
        else:
            # Generate 20 random colors and keep the one most different from prior colors
            best_color = None
            max_min_color_diff = 0.
            for _ in range(20):
                color = _rng.uniform(0., 1., size=3)
                min_color_diff = np.inf
                for existing_color in _block_name_to_color.values():
                    diff = np.sum(np.subtract(color, existing_color)**2)
                    min_color_diff = min(diff, min_color_diff)
                if min_color_diff > max_min_color_diff:
                    best_color = color
                    max_min_color_diff = min_color_diff
        _block_name_to_color[block_name] = best_color

    return _block_name_to_color[block_name]


def block_name_to_image(block_name,  rng_gen):
    
    if block_name not in _block_name_to_class:
        #Assign a class to this block name if not already assigned
        used_classes = set(_block_name_to_class.values())
        available_classes = [c for c in _class_indices_mnist if c not in used_classes]
        if not available_classes:
            raise ValueError("No available classes left to assign.")

        #Corresponding class number based on the block's name (a -> 0, b -> 1, ...)
        chosen_class = ord(block_name[0])-97
        _block_name_to_class[block_name] = chosen_class
    
    #Retrieve and return a random image index from the assigned class
    chosen_class = _block_name_to_class[block_name]
    if not _class_indices_mnist[chosen_class]:
        raise ValueError(f"No more images left in class {chosen_class} for block {block_name}")

    index = rng_gen.choice(_class_indices_mnist[chosen_class])
    _block_name_to_image[block_name] = index 
    return index



def draw_blocks_mnist(ax, block_width, block_height, block_positions, rng_gen):

    for block_name, (x, y) in block_positions.items():
        #color = block_name_to_color(block_name)
        image_mnist = _mnist_images[block_name_to_image(block_name, rng_gen)]
        rect = patches.Rectangle((x,y), block_width, block_height, 
            linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor='none')
        image_artist = ax.imshow(image_mnist, extent=[x, x+block_width, y, y+block_height], cmap='gray')
        #rect = patches.Rectangle((x,y), block_width, block_height, 
        #    linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=color)
        ax.add_patch(rect)

def draw_blocks(ax, block_width, block_height, block_positions):

    for block_name, (x, y) in block_positions.items():
        color = block_name_to_color(block_name)
        rect = patches.Rectangle((x,y), block_width, block_height, 
           linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=color)
        ax.add_patch(rect)

def render(obs, rng_gen, mnist=False, mode='human', close=False):

    width, height = 3.2, 3.2
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(-0.05, width + 0.05),
                                ylim=(-0.05, height + 0.05))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    table_height = height * 0.15
    robot_height = height * 0.1

    piles, holding = get_objects_from_obs(obs, rng_gen)
    block_width, block_height, block_positions = get_block_params(piles, width, height, 
        table_height, robot_height)

    robot_width = block_width * 1.4
    robot_midx = width / 2
    robot_midy = height - robot_height/2

    if mnist:
        #Use images instead of colors
        draw_table(ax, width, table_height)
        draw_blocks_mnist(ax, block_width, block_height, block_positions, rng_gen)
        draw_robot_mnist(ax, robot_width, robot_height, robot_midx, robot_midy, holding,
            block_width, block_height, rng_gen)
        return fig2data(fig), _block_name_to_image, _block_coordinates
    else:
        #Use colors as normal
        draw_table(ax, width, table_height)
        draw_blocks(ax, block_width, block_height, block_positions)
        draw_robot(ax, robot_width, robot_height, robot_midx, robot_midy, holding,
            block_width, block_height)
        return fig2data(fig), _block_name_to_color, _block_coordinates
