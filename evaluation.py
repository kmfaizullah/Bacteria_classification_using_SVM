import numpy as np
import pandas as pd 
from subprocess import check_output
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from pathlib import Path
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
from skimage.transform import resize
import skimage
from sklearn.metrics import classification_report
import os
import cv2

class evaluation:
    
    def individualEvaluation(self,path,resize_height,resize_width,model_name):
        img = cv2.imread(path, cv2.IMREAD_COLOR) 
        img_resized = resize(img,(resize_height, resize_width), anti_aliasing=True, mode='reflect')
        k=img_resized/255
        img=k.flatten()
        flat_data = np.array(img)
        p=img.reshape(1,-1)
        k=model_name.predict(p)
        return k[0] 
    
    def allEvaluation(self,path,height,width,model_name):

        dirs = os.listdir( path )

        flat_data = []
        k=[]
        i=0
        for item in dirs:
            k.append(item)
            img = cv2.imread(path+k[i], cv2.IMREAD_COLOR)
            img_resized = resize(img, (height,width), anti_aliasing=True, mode='reflect')
            normalize=img_resized/255
            flat_data.append(normalize.flatten())
            i=i+1

        flat_data = np.array(flat_data)
        prediction=model_name.predict(flat_data)

        return prediction

