import os
import errno
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


y_train = []
i = 0

with open('./data/list_attr_celeba.txt') as f:
    for line in f:
        a = line.split()
        a = list(a)
        
        if i == 1:
            name = a
        if i >= 2:
            a.pop(0)
            y_train.append(a)
        i += 1

        # Note: only for test, 30 datapoints are chosen
        if i==32:
            break

#print(len(name))
#Black hair, Blond hair, Eyeglasses, Male
# -1 = not black hair
# -1 = not blone hair
# -1 = no glass
# -1 = male
attr = [8, 9, 15, 20]
y_train = np.array(y_train)
y_train = y_train.astype(np.int)
y_mini = y_train[:, attr]
np.save('./data/y_mini_attr', y_mini)

def visualize(data, index):    
    #print(data[index])
    img=mpimg.imread("./data/mini_img/00000" + str(index+1)+ ".png")
    imgplot = plt.imshow(img)
    plt.show()
#visualize(y_mini, 2000)