import os
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from experiment import Experiment
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras import backend as K


def create_experiment_folder(storage_super_folder):
    index = 0
    while os.path.exists(os.path.join(storage_super_folder, f'Experiment_{index}')):
        index += 1
    storage_folder = os.path.join(storage_super_folder, f'Experiment_{index}')
    os.makedirs(storage_folder)
    return storage_folder


def run(dataset_path_index, storage_super_folder, max_distances:list, classifier_types:list, n_runs:list,
        pre_normalizer, inputer, undersampler, split_ratio=0.3 , norm=2, n_jobs_pipeline=-1):

    experiment_folder = create_experiment_folder(storage_super_folder)

    for max_distance in max_distances:
        param_folder = os.path.join(experiment_folder, f'EPS_{max_distance}')
        os.makedirs(param_folder)

        for classifier_type in classifier_types:
            classifier_folder = os.path.join(param_folder, f'{classifier_type}')
            os.makedirs(classifier_folder)

            for run in range(n_runs):
                storage_folder = os.path.join(classifier_folder, f'Run_{run}')
                print(storage_folder)
                os.makedirs(storage_folder)

                experiment = Experiment(classifier_type, dataset_path_index, pre_normalizer, split_ratio,
                                        storage_folder)
                experiment.run_experiment(inputer, undersampler, n_jobs_pipeline, norm, max_distance)


STORAGE_NAMES = ['Attack_results/MLG_Kaggle', 'Attack_results/Vesta']
dataset_path_index = 0 # 0 = MLG, 1 = Vesta
storage_super_folder = STORAGE_NAMES[dataset_path_index]

max_distances = [0.1, 0.25, 0.5, 1, 1000] # 0.1, 0.2, 0.3, 0.5,  1, 10,
classifier_types = ['RF'] #'DNN', 'RF',
n_runs = 5
pre_normalizer = MinMaxScaler()
inputer = SimpleImputer()
undersampler = RandomUnderSampler(sampling_strategy='auto')


run(dataset_path_index, storage_super_folder, max_distances, classifier_types,
    n_runs, pre_normalizer, inputer, undersampler)
K.clear_session()

