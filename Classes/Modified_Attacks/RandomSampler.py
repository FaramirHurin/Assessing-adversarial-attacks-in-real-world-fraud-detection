import copy
import numpy as np
from art.attacks.attack import EvasionAttack
class RandomSamplerAttack:
    def __init__(self, estimator, eps=1, batch_size: int = 64,  norm=2,max_queries = 1000,  targeted=True, classier_type = 'DNN'):
        self.queries_performed = 0
        self.batch_size = batch_size
        self.estimator = estimator
        self.max_queries = max_queries
        self.eps = eps
        self.norm = norm
        self.classier_type = classier_type

    '''
    Make it targeted?
    '''
    def generate(self, x, y=None):
        succcesfull_attacks = 0
        radius = self.eps

        attacks = copy.copy(x)
        for index in range(x.shape[0]):
            positive = 1
            while self.queries_performed < self.max_queries and positive:
                # Generate random array with the same number of columns as x
                step = np.random.randn(x.shape[1])
                step = step / np.linalg.norm(step, ord=2) * radius
                while np.sum(np.abs(step) ** 2) ** (1 / 2) > radius:
                    step = np.random.randn(x.shape[1])
                    step = step / np.linalg.norm(step, ord=2) * radius
                observation = attacks[index, :] + step
                positive = np.round(self._predict_and_count(observation)[0][1])
            if not positive:
                attacks[index] = observation
                succcesfull_attacks += 1

        print (str(succcesfull_attacks) + ' succesfull attacks')
        return attacks

    def _predict_and_count(self, x):
        self.queries_performed += x.shape[0]
        if self.classier_type == 'RF':
            preds =  self.estimator.predict(x.reshape(1, -1))
        else:
            preds =  self.estimator.predict(x.reshape(1, -1), batch_size=self.batch_size)
        return preds



