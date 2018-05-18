import os
import errno
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt


data_dir = os.path.join("./data", "mini_img/")
filenames = os.listdir(data_dir)

X = []
for imge_file in filenames:
    x = plt.imread(data_dir + imge_file)
    # x = imresize()
    X.append(x)

X = np.array(X)

np.save(data_dir + 'X', X)
print (X.shape)

