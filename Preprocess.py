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
import cv2

class Process:
    def __init__(self, data_directory):
        self.container_path = data_directory

    def HyperParameterTuning(self,X_train,y_train,X_test,y_test,cv):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=cv,
                               scoring='%s_macro' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
            
    def load_image_files_modified(self,dimension=(64, 64)):

        image_dir = Path(self.container_path)
    
        folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
        categories = [fo.name for fo in folders]

        descr = "A image classification dataset"
        images = []
        flat_data = []
        target = []
        for i, direc in enumerate(folders):
            for file in direc.iterdir():
                z=str(file)
                img = cv2.imread(z, cv2.IMREAD_COLOR) 
                img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
                normalize=img_resized/255
                flat_data.append(normalize.flatten()) 
                images.append(img_resized)
                target.append(i)
        flat_data = np.array(flat_data)
        target = np.array(target)
        images = np.array(images)

        return Bunch(data=flat_data,
                     target=target,
                     target_names=categories,
                     images=images,
                     DESCR=descr)
