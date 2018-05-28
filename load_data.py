import os
import errno
import numpy as np
import scipy
import scipy.misc
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.pyplot as plt



data_dir = os.path.join("./data", "celebA_img/")
filenames = os.listdir(data_dir)

X_train = []
i = 0
for image_file in filenames:
	if image_file.endswith('.png'):
		x = plt.imread(data_dir + image_file)
		x = imresize(x, (64, 64))
		#X.append(x)
		X_train.append(x)
		# imsave('resize_' + str(i) + '.png', x)
		# i += 1

# X = np.array(X)

# np.save(data_dir + 'X', X)
# X = np.load(data_dir + 'X.npy')
#print (X.shape)
# plt.imshow(X[29])
# plt.show()

