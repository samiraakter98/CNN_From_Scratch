import pandas as pd
import numpy as np
import cv2
import os

filename = 'training-a.csv'
def load_data(filename):
    data = pd.read_csv(filename)
    data = data.loc[:, ['filename', 'digit',]].values
    folder = filename.split('.')[0]
    print(folder)
    x = []
    y = []
    for name in data[:,0]:
        img_dir = os.path.join(folder,name)
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (1,img.shape[0],img.shape[1]))
        x.append(img)
    x = np.array(x)
    y = data[:,1]
    y = np.reshape(y, (y.size,1))

    return (x,y)
x,y = load_data(filename)
print(x.shape)
print(y.shape)