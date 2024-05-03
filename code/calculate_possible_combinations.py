from copy import deepcopy
import itertools




def remove_dups(k):
    k.sort()
    return len(list(k for k,_ in itertools.groupby(k)))



def ways_to_place(available_blocks, configuration):
    total = 0    
    if not available_blocks:
        total_configs.append(configuration)
    else:
        for i, block in enumerate(available_blocks):
            for j, spot in enumerate(configuration):
                if j == block:
                    temp = deepcopy(configuration)
                    temp[j].append(block)
                    
                    temp_blocks = deepcopy(available_blocks)

                    temp_blocks.remove(block)

                    ways_to_place(temp_blocks, temp)
                elif (j == len(configuration)-1 and spot == []):
                    temp = deepcopy(configuration)
                    temp[j].append(block)
                    
                    temp_blocks = deepcopy(available_blocks)

                    temp_blocks.remove(block)

                    ways_to_place(temp_blocks, temp)
                elif j != len(configuration)-1 and spot != []:
                    temp = deepcopy(configuration)
                    temp[j].append(block)
                    
                    temp_blocks = deepcopy(available_blocks)

                    temp_blocks.remove(block)
                    ways_to_place(temp_blocks, temp)

N = 7

values = []
for n in range(1, 7):
    total_configs = []

    ways_to_place([i for i in range(n)], [[] for i in range(n+1)])

    cleaned_list = remove_dups(total_configs)
    values.append(cleaned_list)

import matplotlib.pyplot as plt

x_positions = range(1,7)

# Creating the bar plot
plt.plot(x_positions, values, color='blue')  # You can choose any color

# Adding title and labels
plt.title('Bar Plot of Integers')
plt.xlabel('Index in list')
plt.ylabel('Integer Value')

# Optionally, setting the x-ticks to show indexes starting from 1 or more meaningful labels
plt.xticks(x_positions, [f'Item {i}' for i in x_positions])

# Show the plot
plt.show()