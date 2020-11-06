import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler       # scaling data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split   # splitting data
from sklearn.metrics import accuracy_score             # grading

from sklearn.linear_model import Perceptron

# ******************************************************************************
# * Load data
# ******************************************************************************

df = pd.read_csv('cse575_project/cse575_proj_ax/cse575_proj_data/Processed_Data.csv')

x = df.iloc[:, 0:10]
y = df.iloc[:, 10:]

# x = pd.read_csv('cse575_project/cse575_proj_ax/cse575_proj_data/x_raw.csv')
# y = pd.read_csv('cse575_project/cse575_proj_ax/cse575_proj_data/y_raw.csv')

# ******************************************************************************
# * Create model
# ******************************************************************************

accuracy = []

for i in range(1000):

    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x, y, test_size=0.3)

    stdsc = StandardScaler()
    x_train_std = stdsc.fit_transform(x_train_split)
    x_test_std = stdsc.transform(x_test_split)

    model = LogisticRegression(C=7.1, solver='liblinear', multi_class='auto', random_state=0)
    model.fit(x_train_std, y_train_split.values.ravel())

    accuracy.append(accuracy_score(model.predict(x_test_std), y_test_split))

print(np.mean(accuracy))
