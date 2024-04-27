from Classes.Multiple_attacks_tester import  Multiple_attacks_tester
from Classes.DataSet_Curator import DataSet_Curator
from Classes.Model_traner import ModelTrainer
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from art.estimators.classification import  KerasClassifier as ArtKeras
from sklearn.ensemble import RandomForestClassifier
from art.estimators.classification import SklearnClassifier
import os
import pickle

path_options = ['Classes/Datasets/creditcard_clean.csv', 'Classes/Datasets/vesta_fullDataset_preprocessed.csv']

class Experiment:
    def __init__(self, classifier_type, dataset_path_index, pre_normalizer, split_ratio, STORAGE_FOLDER):
        self.storage_folder = STORAGE_FOLDER
        self.classifier_type = classifier_type
        self.base_classifier = self.__instanciate_base_classifier(classifier_type)
        self.dataset_Curator = DataSet_Curator(path_options[dataset_path_index], pre_normalizer, split_ratio)
        self.param_grid = self.__get_param_grid(self.dataset_Curator)
        X_test, y_test = self.dataset_Curator.get_test_data()
        self._store_datasets(X_test, y_test, STORAGE_FOLDER)
        # Store data

    def __instanciate_base_classifier(self, classifier_type):
        if classifier_type == 'DNN':
            base_classifier = KerasClassifier(build_fn=ModelTrainer.create_keras_model, epochs=50, batch_size=32,
                                              verbose=0)
        elif classifier_type == 'RF':
            base_classifier = RandomForestClassifier()
        return base_classifier

    def run_experiment(self, inputer, undersampler, n_jobs_pipeline=-1, norm=2, max_distance=0.2):

        trained_pipeline = self.__train_pipeline(inputer, undersampler, n_jobs_pipeline)

        self.dataset_Curator.preprocess_data(trained_pipeline)

        classifier = self.__make_classifier_from_pipeline(trained_pipeline)
        best_params = trained_pipeline.best_params_
        classifier_name = self.classifier_type

        attack_name_file_path = os.path.join(self.storage_folder, 'classifier_name.txt')
        with open(attack_name_file_path, 'w') as file:
            file.write(classifier_name)

        attack_params_file_path = os.path.join(self.storage_folder, 'best_params.pkl')
        with open(attack_params_file_path, 'wb') as file:
            pickle.dump(best_params, file)


        X_test_new, _ = self.dataset_Curator.get_test_data_clean()
        predictions = classifier.predict(X_test_new)

        predictions_file_path = os.path.join(self.storage_folder, 'predictions.npy')
        ytest_file_path = os.path.join(self.storage_folder, 'y_test.npy')

        np.save(predictions_file_path, predictions)
        y_test = self.dataset_Curator.get_test_data_clean()[1]
        np.save(ytest_file_path, y_test)


        attacks_storage_folder = os.path.join(self.storage_folder, 'attacks')
        os.makedirs(attacks_storage_folder)
        attacks = self.__run_attacks(classifier, norm, max_distance, attacks_storage_folder)

        return attacks, classifier

    def test_only_classification(self,inputer, undersampler, n_jobs_pipeline=-1):
        trained_pipeline = self.__train_pipeline(inputer, undersampler, n_jobs_pipeline)
        self.dataset_Curator.preprocess_data(trained_pipeline)
        classifier = self.__make_classifier_from_pipeline(trained_pipeline)
        X_test_new, _ = self.dataset_Curator.get_test_data_clean()
        predictions = classifier.predict(X_test_new)
        return self.dataset_Curator.get_test_data_clean()[1], predictions

    def __train_pipeline(self, inputer, undersampler, n_jobs_pipeline):
        model_Trainer = ModelTrainer(self.classifier_type, self.param_grid, self.dataset_Curator, inputer, undersampler)
        trained_pipeline = model_Trainer.train_pipeline(n_jobs=n_jobs_pipeline)
        return trained_pipeline

    def __make_classifier_from_pipeline(self, trained_pipeline):
        best_params = trained_pipeline.best_params_
        filtered_params = {key.replace('classifier__', ''): value for key, value in best_params.items() if
                           key.startswith('classifier__')}
        X_train_clean, y_train_clean = self.dataset_Curator.get_training_data_clean()
        print(y_train_clean.shape)
        if self.classifier_type == 'DNN':
            filtered_params['output_shape'] = 2
            classifier = ArtKeras(ModelTrainer.create_keras_model(**filtered_params))
            classifier.fit(X_train_clean, y_train_clean, batch_size=32, nb_epochs=20)

        else:
            classifier = SklearnClassifier(RandomForestClassifier(**filtered_params))
            classifier.fit(X_train_clean, y_train_clean)

        return classifier

    def __run_attacks(self, classifier, norm, max_distance,  folder):
        attacks_creator = Multiple_attacks_tester( self.dataset_Curator, classifier, norm, max_distance)
        print('Run Attacks please :) ')
        attacks = attacks_creator.run_attacks( folder)
        return attacks  # Returns dictionary of attacks. Can we make pass also the hyperparams?

    def __get_param_grid(self, dataset_Curator, undersampling_fator = None):
        X_train, y_train = dataset_Curator.get_training_data()

        if self.classifier_type == 'DNN':
            param_grid = {
            'sampler__sampling_strategy': [0.05, 0.1],  # 0.005, 0.01, 0.1, 0.25,
            'classifier__optimizer': ['adam'],  # , 'sgd'
            'classifier__activation': ['relu'],  # , 'tanh'
            'classifier__input_shape': [X_train.shape[1], ],
            'classifier__hidden_units': [[16, 8], [32, 32]],  # [128, 64] , [500, 500] , [128, 128]
            'imputer__strategy': ['most_frequent'],  #median', 'mean',
            #'classifier__input_regularizer': [None], #, 0.001 , 0.01
            'classifier__dropout_rate' : [0, 0.1, 0.25]
            }
        elif self.classifier_type == 'RF':
            param_grid = {
                'sampler__sampling_strategy': [0.05,  0.1],  #
                'imputer__strategy': [ 'most_frequent'],  # , 'median', 'median',
                'classifier__n_estimators': [ 50, 100, 500],  # 1000, etc.
                'classifier__criterion': ["gini"], #, "entropy"
                'classifier__max_depth': [None, 20],
                # , 'classifier__min_samples_split': []
            }

        # Allows for testing for a given undersampling factor
        if undersampling_fator is not None:
            param_grid['sampler__sampling_strategy'] = undersampling_fator

        return param_grid


    @staticmethod
    def _store_datasets(X_test, y_test, STORAGE_FOLDER):
        X_test_path = os.path.join(STORAGE_FOLDER, 'X_test.npy')
        y_test_path = os.path.join(STORAGE_FOLDER, 'y_test.npy')

        with open(X_test_path, 'wb') as f:
            np.save(f, X_test)

        with open(y_test_path, 'wb') as f:
            np.save(f, y_test)


