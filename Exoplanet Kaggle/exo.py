import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy import ndimage
import matplotlib
from scipy import stats
import warnings
import tensorflow
from sklearn import svm
from sklearn import metrics
import sklearn
import sklearn.model_selection
import keras
warnings.filterwarnings('ignore')


import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

#print(extrain.LABEL.value_counts())
#print(extest.LABEL.value_counts())
import imblearn

from imblearn.over_sampling import SMOTE

#Reading in the data
data=pd.read_csv("exoTest.csv",sep=",")

#Extracting the labels from the dataset
predict="LABEL"
x = np.array(data.drop(['LABEL'], 1))
y = np.array(data['LABEL'])

print(data.head())
print(x)
print(y)


#splitting the remaining data into training sets and test sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
print(x_train)
print(y_train)
#decoding the 1's and 2's into their actual meanings
classes = ['Exoplanet present','Exoplanet not present']

#fitting the model
clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(y_test[:100])
print(y_pred[:100])

extrain = pd.read_csv('exoTrain.csv')
extest = pd.read_csv('exoTest.csv')
#print(extrain.head())

i = 13
flux1 = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
time = np.arange(len(flux1)) * (36.0/60.0)  # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Original flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Flux, e-/s')
plt.xlabel('Time, hours')
#plt.plot(time, flux1)
#plt.show()

i = 13
flux2 = ndimage.filters.gaussian_filter(flux1, sigma=10)
time = np.arange(len(flux2)) * (36.0/60.0) # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Smoothed flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Flux, e-/s')
plt.xlabel('Time, hours')
#plt.plot(time, flux2)
#plt.show()

flux3 = flux1-flux2
time = np.arange(len(flux3)) * (36.0/60.0)  # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Detrended flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Flux, e-/s')
plt.xlabel('Time, hours')
plt.plot(time, flux3)
#plt.show()

flux3normalised = (flux3-np.mean(flux3))/(np.max(flux3)-np.min(flux3))
time = np.arange(len(flux3normalised)) * (36.0/60.0)  # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Normalised, Detrended flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Normalised Flux, e-/s')
plt.xlabel('Time, hours')
plt.plot(time, flux3)
plt.show()

def detrender_normalizer(X):
    flux1=X
    flux2=ndimage.filters.gaussian_filter(flux1,sigma=10)
    flux3 = flux1-flux2
    flux3normalised=(flux3-np.mean(flux3))/(np.max(flux3)-np.min(flux3))
    return flux3normalised

extrain.iloc[:,1:] = extrain.iloc[:,1:].apply(detrender_normalizer, axis=1)
extest.iloc[:,1:] = extest.iloc[:,1:].apply(detrender_normalizer, axis=1)

flux1 = extrain[extrain.LABEL==2].drop('LABEL', axis=1).iloc[i,:]
flux1=flux1.reset_index(drop=True)
time = np.arange(len(flux3normalised)) * (36.0/60.0)  # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Processed flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Normalised Flux, e-/s')
plt.xlabel('Time, hours')
plt.plot(time, flux1)
plt.show()

for i in [0, 9, 14, 19, 24, 29]:
    flux = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:] #.drop('LABEL', axis=1).iloc[i,:]
    time = np.arange(len(flux)) * (36.0/60.0) #time in hours
    plt.figure(figsize=(15,5))
    plt.title('Flux of star {} with confirmed exoplanets'.format(i+1))
    plt.ylabel('Flux, e-/s')
    plt.xlabel('Time, hours')
    plt.plot(time, flux)
    plt.show()

df = pd.read_csv('exoTrain.csv', index_col=0)
def reduce_upper_outliers(df, reduce = 0.01, half_width=4):
    length = len(df.iloc[0,:])
    remove = int(length*reduce)
    for i in df.index.values:
        values = df.loc[i,:]
        sorted_values = values.sort_values(ascending = False)
        # print(sorted_values[:30])
        for j in range(remove):
            idx = sorted_values.index[j]
            # print(idx)
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            # print(idx,idx_num)
            for k in range(2 * half_width + 1):
                idx2 = idx_num + k - half_width
                if idx2 < 1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.' + str(idx2)]  # corrected from 'FLUX-' to 'FLUX.'
                count += 1
            new_val /= count  # count will always be positive here
            # print(new_val)
            if new_val < values[idx]:  # just in case there's a few persistently high adjacent values
                df.at[i, idx]= new_val

    return df

extrain.iloc[:, 1:] = reduce_upper_outliers(extrain.iloc[:, 1:])
extest.iloc[:, 1:] = reduce_upper_outliers(extest.iloc[:, 1:])

i = 13
flux1 = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
flux1 = flux1.reset_index(drop=True)
time = np.arange(len(flux1)) * (36.0/60.0) # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Processed flux of star {} with confirmed exoplanets (removed upper outliers)'.format(i+1))
plt.ylabel('Normalized flux')
plt.xlabel('Time, hours')
plt.plot(time, flux1)
plt.show()


def model_evaluator(X, y, model, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits)

    bootstrapped_accuracies = list()
    bootstrapped_precisions = list()
    bootstrapped_recalls = list()
    bootstrapped_f1s = list()

    SMOTE_accuracies = list()
    SMOTE_precisions = list()
    SMOTE_recalls = list()
    SMOTE_f1s = list()

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        df_train = X_train.join(y_train)
        df_planet = df_train[df_train.LABEL == 2].reset_index(drop=True)
        df_noplanet = df_train[df_train.LABEL == 1].reset_index(drop=True)
        df_boot = df_noplanet
        index = np.arange(0, df_planet.shape[0])
        temp_index = np.random.choice(index, size=df_noplanet.shape[0])
        df_boot = df_boot.append(df_planet.iloc[temp_index])

        df_boot = df_boot.reset_index(drop=True)
        X_train_boot = df_boot.drop('LABEL', axis=1)
        y_train_boot = df_boot.LABEL

        est_boot = model.fit(X_train_boot, y_train_boot)
        y_test_pred = est_boot.predict(X_test)

        bootstrapped_accuracies.append(accuracy_score(y_test, y_test_pred))
        bootstrapped_precisions.append(precision_score(y_test, y_test_pred, pos_label=2))
        bootstrapped_recalls.append(recall_score(y_test, y_test_pred, pos_label=2))
        bootstrapped_f1s.append(f1_score(y_test, y_test_pred, pos_label=2))
        sm = SMOTE(ratio=1.0)
        X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

        est_sm = model.fit(X_train_sm, y_train_sm)
        y_test_pred = est_sm.predict(X_test)

        SMOTE_accuracies.append(accuracy_score(y_test, y_test_pred))
        SMOTE_precisions.append(precision_score(y_test, y_test_pred, pos_label=2))
        SMOTE_recalls.append(recall_score(y_test, y_test_pred, pos_label=2))
        SMOTE_f1s.append(f1_score(y_test, y_test_pred, pos_label=2))

    print('\t\t\t Bootstrapped \t SMOTE')
    print("Average Accuracy:\t", "{:0.10f}".format(np.mean(bootstrapped_accuracies)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_accuracies)))
    print("Average Precision:\t", "{:0.10f}".format(np.mean(bootstrapped_precisions)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_precisions)))
    print("Average Recall:\t\t", "{:0.10f}".format(np.mean(bootstrapped_recalls)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_recalls)))
    print("Average F1:\t\t", "{:0.10f}".format(np.mean(bootstrapped_f1s)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_f1s)))

extrain_raw=pd.read_csv('exoTrain.csv')
X_raw=extrain_raw.drop('LABEL', axis=1)
y_raw =extrain_raw.LABEL
model_evaluator(X_raw, y_raw, LinearSVC)
