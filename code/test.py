import pddlgym
import numpy as np
import imageio
import matplotlib.pyplot as plt
def create_random_environment():
    # Load the Sokoban environment from PDDLGym

    env = pddlgym.make("PDDLEnvBlocks-v0", seed = 1)
    obs, _ = env.reset()

    img = env.render()
    imageio.imsave("testimage.png", img)


    '''for k in range(30):

        action = env.action_space.sample(obs)
        state, reward, done, debug_info = env.step(action)
        img = env.render()

        print(state.literals)

        plt.imshow(img, cmap='gray')  # Use cmap='gray' for grayscale images
        plt.show()'''


    # Your logic here to interact with the environment
    # For example, applying a random action:
    '''action = env.action_space.sample()
     next_state, reward, done, info = env.step(action)
     env.render()
    img = env.render()
    imageio.imsave("testimage.png", img)
    print(img.shape)'''

if __name__ == "__main__":
    
    create_random_environment()
