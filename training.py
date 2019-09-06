from Preprocess import Process
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics, datasets
from sklearn.metrics import classification_report
import pickle 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from evaluation import evaluation
import os
import skimage
from skimage.transform import resize
import numpy as np
import cv2
from pathlib import Path
from sklearn.utils import Bunch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC



class training:
    
     def __init__(self,path,test_size,random_state):
        self.process=Process(path)
        self.evaluation=evaluation()
        self.result=self.process.load_image_files_modified()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.result.data, self.result.target, test_size=test_size,random_state=random_state)
    
     def train(self,validation_size):
        self.process.HyperParameterTuning(self.X_train,self.y_train,self.X_test,self.y_test,validation_size)
        classifier = svm.SVC(C=1000,gamma=0.001,kernel='linear',decision_function_shape='ov0',probability=False)
        classifier.fit(self.X_train,self.y_train)
        print('Accuracy of training set: {:.2f}'.format(classifier.score(self.X_test, self.y_test)))
        model = pickle.dumps(classifier) #saving model
        return classifier
    
        
    
