import os
import errno
import numpy as np

y_train = []
i = 0

with open('list_attr_celeba.txt') as f:
    for line in f:
        a = line.split()
        a = list(a)
        a.pop(0)
        if i == 1:
            name = a
        if i >= 2:
            y_train.append(a)
        i += 1
        if i==32:
            break

#Black hair, Blond hair, Eyeglasses, Male
attr = [7, 8, 14, 19]
y_train = np.array(y_train)
y_train = y_train.astype(np.int)

print(y_train[0])
y_mini = [y_train[index] for index in attr]
print(y_mini)
