from .Attacks_creator import Attacks_creator
import pickle as pk
import random
import os
from art.attacks.evasion import CarliniL2Method, DeepFool, FastGradientMethod, ProjectedGradientDescent
from .Modified_Attacks.Boundary_new import Boundary_new
from .Modified_Attacks.HopSkipJump_new import HopSkipJump_new
from .Modified_Attacks.RandomSampler import  RandomSamplerAttack
from .Modified_Attacks.ZOO_new import  ZooAttack_new


class Multiple_attacks_tester:
    def __init__(self, dataset_curator, classifier, norm, max_distance:float):
        self.classifier_type = type(classifier).__name__
        print(self.classifier_type)
        self.classifier = classifier
        self.norm = norm
        self.max_distance = max_distance
        self.attack_classes_dict = self.__create_attacks_dictionary()
        self.number_of_stored_files = 0
        self.X_test, self.y_test = dataset_curator.get_test_data_clean()

        return

    def __create_attacks_dictionary(self):
        if self.classifier_type == 'KerasClassifier':

            with open(str(os.path.dirname(os.path.abspath(__file__)))+'/NN_attacks_dictionary.txt', 'r') as file:
                file_contents = file.read()
        elif self.classifier_type == 'ScikitlearnRandomForestClassifier':
            print(self.classifier_type)
            with open(str(os.path.dirname(os.path.abspath(__file__)))+'/RF_attacks_dictionary.txt', 'r') as file:
                file_contents = file.read()
        attack_dictionary = eval(file_contents)
        return attack_dictionary

    def run_attacks(self, Attacks_folder):
        attack_data_dict = {}
        parameters = {}
        predictions = {}
        queries_number = {}
        attacks_names = self.attack_classes_dict.keys()
        attack_data_dict['Genuine'] = self.X_test
        for attack_name in attacks_names:
            parameters[attack_name] = self.__sample_params(attack_name)
            attack_instance = self.__create_attack_instance(attack_name, parameters[attack_name])
            print(parameters[attack_name])
            attack_data_dict[attack_name] = self.__run_single_attack(attack_instance)
            predictions[attack_name] = self.classifier.predict(attack_data_dict[attack_name])[:,1]
            if attack_name == 'BoundaryAttack' or attack_name == 'HopSkipJump' or attack_name == 'ZooAttack' or attack_name == 'RandomSamplerAttack':
                queries_number[attack_name] = attack_instance.queries_performed
            DEBUG = 0
        self._store_attacks(Attacks_folder, parameters, attack_data_dict, predictions, queries_number)


    def __create_attack_instance(self, attack_name, attack_parameters):
        attack_class = self.attack_classes_dict[attack_name]['class_type']
        if attack_name in ['CarliniLInfMethod', 'CarliniL2Method', 'DeepFool', 'ZooAttack', 'BoundaryAttack']:
            attack_instance = attack_class(self.classifier, **attack_parameters)
        elif attack_name == 'RandomSamplerAttack':
            attack_instance = attack_class(self.classifier, **attack_parameters)
        else:
            attack_instance = attack_class(self.classifier, **attack_parameters, norm=self.norm)
        if attack_name == 'FastGradientMethod' or attack_name == 'ProjectedGradientDescent':
            attack_instance.set_params(eps= self.max_distance)
            print(attack_instance.eps)
        if attack_name == 'RandomSamplerAttack':
            attack_instance.eps = self.max_distance
        return attack_instance

    def __sample_params(self, attack_name):
        random_params = {param: random.choice(values)
                             for param, values in self.attack_classes_dict[attack_name]['hyperparameters'].items()}
        return random_params

    def __run_single_attack(self, attack_instance):
        attack_creator = Attacks_creator( attack_instance, self.classifier, self.X_test, self.y_test, self.norm, self.max_distance)
        X_with_feasible_atacks, y_test =  attack_creator.compute_attack()
        return X_with_feasible_atacks

    @staticmethod
    def _store_attacks(Attacks_folder,  parameters, attack_data_dict, predictions, queries_number):
        parameters_path = os.path.join(Attacks_folder, 'parameters.pkl')
        data_path = os.path.join(Attacks_folder, 'attack_data.pkl')
        predictions_path = os.path.join(Attacks_folder, 'attack_predictions.pkl')
        queries_path =  os.path.join(Attacks_folder, 'queries_number.pkl')
        print(queries_number)
        with open(parameters_path, 'wb') as file:
            pk.dump(parameters, file)
        with open(data_path, 'wb') as file:
            pk.dump(attack_data_dict, file)
        with open(predictions_path, 'wb') as file:
            pk.dump(predictions, file)
        with open(queries_path, 'wb') as file:
            pk.dump(queries_number, file)

