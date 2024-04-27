from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
import numpy as np

class DataSet_Curator:
    def __init__(self, path, normalizer, split_ratio):
        dataset = pd.read_csv(path)
        self.X = dataset.iloc[:, :-1]
        self.y =dataset.iloc[:, -1]
        self.__normalize_data(normalizer)
        self.__train_test_split(split_ratio)

        self.has_preprocessed_data = False

    def __normalize_data(self, pre_normalizer:MinMaxScaler):
        pre_normalizer.fit_transform(self.X)

    def __train_test_split(self, split_ratio):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42,
                                                                                test_size=split_ratio)


    def preprocess_data(self, trained_pipeline:GridSearchCV):
        X_train, y_train = self.get_training_data()
        X_test, y_test = self.get_test_data()

        imputer: SimpleImputer = trained_pipeline.best_estimator_[0]
        undersampler: RandomUnderSampler = trained_pipeline.best_estimator_[1]

        X_train_filled = imputer.transform(X_train)
        X_test_filled = imputer.transform(X_test)

        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_filled, y_train)

        self.X_train_new,  self.y_train_new, self.X_test_new, self.y_test_new =\
            X_train_resampled, y_train_resampled, X_test_filled, y_test

        self.has_preprocessed_data = True

    def get_training_data(self):
        return self.X_train.values, self.y_train.values

    def get_test_data(self):
        return self.X_test.values, self.y_test.values

    def get_input_shape(self):
        return self.X_train.shape[1]

    def get_training_data_clean(self):
        if not self.has_preprocessed_data:
            print('You are trying to access preprocessed data, but preprocessing has not happened yet')
            raise Exception
        return self.X_train_new,np.array([np.ones(self.y_train_new.shape) - self.y_train_new, self.y_train_new]).T

    def get_test_data_clean(self):
        if not self.has_preprocessed_data:
            print('You are trying to access preprocessed data, but preprocessing has not happened yet')
            raise Exception
        return self.X_test_new, np.array([np.ones(self.y_test_new.shape) - self.y_test_new, self.y_test_new]).T


    # Add the pre


