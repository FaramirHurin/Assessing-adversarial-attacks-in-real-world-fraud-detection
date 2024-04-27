import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from imblearn.pipeline import Pipeline as ImbPipeline


class ModelTrainer:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.pipeline = self.build_pipeline()

    def build_pipeline(self) -> ImbPipeline:
        # Define the pipeline
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', RandomOverSampler()),  # You can replace this with other sampling strategies
            ('classifier', self.build_sequential_model())
        ])
        return pipeline

    def build_sequential_model(self) -> KerasClassifier:
        # Define a function that creates the Keras model
        def create_model():
            model = Sequential()
            model.add(Dense(64, activation='relu', input_dim=1))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

    def train_pipeline(self, epochs: int = 10, batch_size: int = 32, param_grid: dict = None, cv: int = 5) -> None:
        # Specify a scoring metric (e.g., accuracy)
        scoring_metric = 'roc_auc'

        # Use GridSearchCV to find the best parameters
        grid_search = GridSearchCV(self.pipeline, param_grid=param_grid, cv=cv, scoring=scoring_metric)
        grid_search.fit(self.X_train, self.y_train)

        # Get the best parameters and build the final pipeline
        best_params = grid_search.best_params_
        self.pipeline.set_params(**best_params)

        # Transform y_train to a bidimensional array
        self.y_train = self.y_train.values.reshape(-1, 1)

        # Fit the final classifier with the best parameters
        self.pipeline.fit(self.X_train.values.reshape(-1, 1), self.y_train, classifier__epochs=epochs,
                          classifier__batch_size=batch_size)
    def preprocess_data(self, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
        # Preprocess X_test and y_test using the pipeline
        X_test_preprocessed = self.pipeline[:-1].transform(X_test.values.reshape(-1, 1))
        y_test_preprocessed = self.pipeline[:-1].transform(y_test.values.reshape(-1, 1))

        # Transform y_test to a bidimensional array
        y_test_preprocessed = y_test_preprocessed.reshape(-1, 1)

        return X_test_preprocessed, y_test_preprocessed

    def evaluate_model(self, X_test_preprocessed: np.ndarray, y_test_preprocessed: np.ndarray) -> float:
        # Predict and evaluate the model
        y_pred = self.pipeline.predict(X_test_preprocessed)
        accuracy = accuracy_score(y_test_preprocessed, y_pred)
        return accuracy

