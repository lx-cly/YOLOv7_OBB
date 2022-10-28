#dataset info
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import cv2
import numpy as np
filename = '../dataset/dataset_demo/images/P0032.png'
print('filename:', filename)
img = cv2.imread(filename)
plt.imshow(img)
plt.axis('off')
r = 5
point = []
ax = plt.gca()
ax.set_autoscale_on(False)
c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
circle = Circle((point[0], point[1]), r)
p = PatchCollection(circle, facecolors='red')
ax.add_collection(p)
plt.show()
# path = '/6TB/lanxin/dota_1024/train/images'
# files1 = os.listdir(path)
# print(len(files1))
# path = '/6TB/lanxin/dota_1024/val/labelTxt'
# files2 = os.listdir(path)
# print(len(files2))
# for file in files2:
# 	f = file.split('.')[0]
# 	ft = f + '.png'
# 	if ft in files1:
# 		continue
# 	else:
# 		print(f)