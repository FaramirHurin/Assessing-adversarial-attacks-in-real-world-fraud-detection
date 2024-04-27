import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from .DataSet_Curator import  DataSet_Curator
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer, ActivityRegularization
from keras.wrappers.scikit_learn import KerasClassifier
from .loss_functions import compute_losses
from imblearn.pipeline import make_pipeline
from tensorflow.keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    def __init__(self, classifier_type:str, grid:dict,  dataset_Curator:DataSet_Curator,
                 imputer:SimpleImputer, undersampler:RandomUnderSampler):
        self.imputer_basic = imputer
        self.undersampler_basic = undersampler
        self.classifier_basic = self.__instanciate_classifier(classifier_type)
        self.classifier_type = classifier_type
        self.grid = grid
        self.dataset_curator = dataset_Curator

    def __instanciate_classifier(self, classifier_type:str):
        if classifier_type == 'DNN':
            classifier_basic = KerasClassifier(build_fn=self.create_keras_model, epochs=20, batch_size=32, verbose=0)
        elif classifier_type == 'RF':
            classifier_basic = RandomForestClassifier()
        return classifier_basic

    @staticmethod
    def create_keras_model(hidden_units:list, activation:str, optimizer:str, input_shape,
                           output_shape=1, dropout_rate=0):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))

        for units in hidden_units:
            model.add(Dense(units=units, activation=activation))
            model.add(Dropout(dropout_rate))

        model.add(Dense(units=output_shape, activation='softmax'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def create_model(units, dropout_rate, optimizer='adam'):
        model = Sequential()
        model.add(keras.Input(shape=(30,)))
        model.add(Dense(units * 2, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train_pipeline(self, n_jobs=1, cv=2):
        X_train, y_train = self.dataset_curator.get_training_data()
        pipeline = self.__make_pipeline()
        trained_pipeline = GridSearchCV(pipeline, param_grid=self.grid, cv=cv, scoring='roc_auc',
                                             n_jobs=n_jobs)
        trained_pipeline.fit(X_train, y_train)
        return trained_pipeline

    def __make_pipeline(self):
        imputer = ('imputer', self.imputer_basic)
        sampler = ('sampler', self.undersampler_basic)
        classifier = ('classifier', self.classifier_basic)
        pipeline = ImbPipeline(steps = [imputer, sampler, classifier])
        return pipeline


        # Create the imblearn pipeline with custom names
        imblearn_pipeline = make_pipeline(imputer, sampler, classifier)


#1 Change parameters
# Do we need test set clean?
