import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def training_data():
    groups_folder_path = './cnn_sample/'

    categories = ["flower", "tomato", "dog", "cat"]

    num_classes = len(categories)

    image_w = 256
    image_h = 256

    X = []
    Y = []
    
    for idex, categorie in enumerate(categories):
        label = idex
        image_dir = groups_folder_path + categorie + '/train/'
    
        for top, dir, f in os.walk(image_dir):
            for filename in f:
                img = cv2.imread(image_dir+filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
                X.append(img)
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    xy = (X_train, X_test, Y_train, Y_test)
    
    np.save("./img_data.npy", xy)
    print("DONE")

"""
X = []
Y = []
  
for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + '/train/'
  
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            img = cv2.imread(image_dir+filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, None, fx=image_w/img.shape[0], fy=image_h/img.shape[1])
            X.append(img/255)
            Y.append(label)
 
# for i in range(990, 1005):
#     cv2.imshow("asdf",X[i])
#     print(Y[i])
#     cv2.waitKey(0)


X = np.array(X)
Y = np.array(Y)
"""


