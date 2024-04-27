import copy

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from RandomSampler import  RandomSamplerAttack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

dataset = pd.read_csv('../Datasets/creditcard.csv')

normalizer = MinMaxScaler()
dataset = normalizer.fit_transform(dataset)

X = dataset[:, :-1]
y = np.array([ np.ones(dataset.shape[0]) - dataset[:, -1],dataset[:, -1]]).T

X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForestClassifier(10)
rf.fit(X_train, y_train)
predictions_genuine = rf.predict(X_test)[:,1]
f1_clean = f1_score(y_test[:,1], np.round(predictions_genuine))

attack = RandomSamplerAttack(rf, classier_type='RF', eps=10, max_queries=10000)
attacked_df = copy.copy(X_test)
to_attack = X_test[(y_test==1)[:,1]]
print(to_attack.shape[0])
assert to_attack.shape[0] < 300
attack_data = attack.generate(to_attack)
attacked_df[(y_test==1)[:,1]] = attack_data
predictions_attack = rf.predict(attacked_df)[:,1]
f1_atk = f1_score(y_test[:,1], np.round(predictions_attack))

print(f1_clean, f1_atk)

