# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:21:26 2018

@author: hassanyf
"""
from __future__ import print_function
from myo.utils import TimeInterval
import myo
import sys
import pandas as pd

# Implementing DeviceListener for EMG. 
class Listener(myo.DeviceListener):

    def __init__(self):
        self.interval = TimeInterval(None, 0.05)
        self.orientation = None
        self.pose = myo.Pose.rest
        self.emg_enabled = False
        self.locked = False
        self.emg = None

    def output(self):
        if not self.interval.check_and_reset():
            return

        parts = []

        if self.emg:
            for comp in self.emg:
                parts.append(str(comp).ljust(5))
        print('\r' + ','.join('{}'.format(p) for p in parts), end='')
        sys.stdout.flush()

    def on_connected(self, event):
        event.device.request_rssi()

    def on_pose(self, event):
        event.device.stream_emg(True)
        self.emg_enabled = True
        self.emg = None
        self.output()

    def on_emg(self, event):
        self.emg = event.emg
        self.output()


dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, [0,1,2,3,4,5,6,7]].values
y = dataset.iloc[:, 8].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)
y=y.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier =RandomForestClassifier(n_estimators =500,criterion='entropy',random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

# Displaying EMG data
if __name__ == '__main__':
    myo.init()
    hub = myo.Hub()
    listener = Listener()
    while hub.run(listener.on_event, 500):
        pass