import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import pickle

def HDD_prediction(x_test):
    filename = "./HDD_classification_model.h5"
    model = pickle.load(open(filename, 'rb'))

    res = model.predict(x_test)
    return res