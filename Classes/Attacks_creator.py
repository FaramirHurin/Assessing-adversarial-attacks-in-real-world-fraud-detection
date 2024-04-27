import numpy as np
import pandas as pd
from art.estimators.classification import  KerasClassifier as ArtKeras
from copy import copy

class Attacks_creator:
    def __init__(self, attack_instance, classifier, X_test, y_test, norm, max_distance):
        self.attack_instance = attack_instance
        self.classifier = classifier
        self.X_test = X_test
        self.y_test = y_test
        self.norm = norm
        self.max_distance = max_distance
        return

    def compute_attack(self):
        selected_indices = self.__select_target_indices()
        selected_data_X = self.X_test[selected_indices, :]
        targets = np.array([np.ones(selected_data_X.shape[0]), np.zeros(selected_data_X.shape[0])]).T
        generated_data: np.ndarray = self.attack_instance.generate(selected_data_X, targets)
        X_with_feasible_attacks = self.__filter_distant_attacks(generated_data, selected_data_X, selected_indices)
        return X_with_feasible_attacks, self.y_test

    def __select_target_indices(self):
        predicted_probabilities = self.classifier.predict(self.X_test)
        classified_as_positive = np.where(predicted_probabilities[:, 1] > predicted_probabilities[:, 0])[0]
        print('Len of classified as positive is ' + str(len(classified_as_positive)) + ' out of ' + str(len(classified_as_positive)))
        true_positive = np.where(self.y_test[:, 1] == 1)[0]

        selected_indices = np.intersect1d(classified_as_positive, true_positive)
        print('Len of selected indices is ' + str(len(selected_indices)) + ' out of ' + str(len(true_positive)))
        return selected_indices

    def __filter_distant_attacks(self, generated_data,  selected_data_X, selected_indices):
        differences = generated_data - selected_data_X
        for row in range(generated_data.shape[0]):
            if np.sum(np.abs(differences[row, :]) ** self.norm) ** (1 / self.norm) > self.max_distance:
                generated_data[row, :] = selected_data_X[row, :]

        X_with_feasible_attacks = copy(self.X_test)
        X_with_feasible_attacks[selected_indices, :] = generated_data

        return X_with_feasible_attacks



    '''
    def select_target_data(self):
        selected_indices = self.select_target_indices()
        selected_data_X = self.X_test.loc[selected_indices, :]
        selected_data_y = self.y_test.loc[selected_indices]
        return selected_data_X, selected_data_y

    def filter_far_attacks_indices(self, generated_data: np.ndarray, selected_data_X: np.ndarray):
        differences: np.ndarray = generated_data - selected_data_X
        distance_norms:np.ndarray = self.compute_distance_norms(differences)
        replace_indices = distance_norms > self.max_distance
        print('Replace index are ' + str(replace_indices))
        filtered_generated_data = generated_data
        filtered_generated_data[replace_indices] = selected_data_X[replace_indices]
        assert filtered_generated_data.shape == generated_data.shape
        return filtered_generated_data

    def compute_attack(self):
        selected_data_X, selected_data_y = self.select_target_data()
        generated_data = self.attack_instance.generate(selected_data_X.values)
        generated_data_df = pd.DataFrame(generated_data, index=selected_data_X.index, columns=selected_data_X.columns)
        attacks_to_keep = self.filter_far_attacks_indices(generated_data_df, selected_data_X)
        X_with_feasible_atacks = self.X_test
        X_with_feasible_atacks.loc[attacks_to_keep.index, :] = generated_data
        return X_with_feasible_atacks, self.y_test

    def compute_distance_norms(self, differences):
        distance_norms = np.zeros(differences.shape[0])
        for line in range(differences.shape[0]):
            distance_norms[line] = np.sum(np.abs(differences[line, :]) ** self.norm) ** (1 / self.norm)
        return distance_norms
    '''