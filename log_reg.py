import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler       # scaling data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split   # splitting data
from sklearn.metrics import accuracy_score             # grading
from sklearn.metrics import f1_score             # grading

from sklearn.linear_model import Perceptron

# ******************************************************************************
# * Load data
# ******************************************************************************

df = pd.read_csv('cse575_project/cse575_proj_ax/cse575_proj_data/output.csv')

x = df.iloc[:, 0:25]
y = df.iloc[:, 25:]

# x = pd.read_csv('cse575_project/cse575_proj_ax/cse575_proj_data/x_raw.csv')
# y = pd.read_csv('cse575_project/cse575_proj_ax/cse575_proj_data/y_raw.csv')

# ******************************************************************************
# * Create model
# ******************************************************************************

# accuracy = []
# macro_f1 = []
# micro_f1 = []
# weighted_f1 = []

for i in range(10):

    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x, y, test_size=0.3)

    stdsc = StandardScaler()
    x_train_std = stdsc.fit_transform(x_train_split)
    x_test_std = stdsc.transform(x_test_split)

    model = LogisticRegression(C=7.1, solver='liblinear', multi_class='auto', random_state=0)
    model.fit(x_train_std, y_train_split)

    y_pred = model.predict(x_test_std)

    # y_pred = []
    # for j in y_pred_inv:
    #     if int(j) == 0:
    #         y_pred.append(1)
    #     else:
    #         y_pred.append(0)

    print(f'Run {i}')
    print('*************************************')

    print(f'Accuracy:       {accuracy_score(y_pred, y_test_split)}')
    print(f'Macro F1:       {f1_score(y_test_split, y_pred, average="macro")}')
    print(f'Micro F1:       {f1_score(y_test_split, y_pred, average="micro")}')
    print(f'Weighted F1:    {f1_score(y_test_split, y_pred, average="weighted")}')

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    print(int(y_pred[0]))
    print(int(y_test_split.iloc[0].values))

    for j in range(y_test_split.shape[0]):
        if int(y_test_split.iloc[j].values) == int(y_pred[j]):
            if int(y_pred[j]) == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if int(y_pred[j]) == 1:
                false_pos += 1
            else:
                false_neg += 1

    print(f'False +:        {false_pos}')
    print(f'False -:        {false_neg}')
    print(f'True  +:        {true_pos}')
    print(f'True  -:        {true_neg}')

    print()

    # accuracy.append(accuracy_score(y_pred, y_test_split))

    # macro_f1.append(f1_score(y_test_split, y_pred, average='macro'))
    # micro_f1.append(f1_score(y_test_split, y_pred, average='micro'))
    # weighted_f1.append(f1_score(y_test_split, y_pred, average='weighted'))


# print(np.mean(accuracy))
# print(np.mean(macro_f1))
# print(np.mean(micro_f1))
