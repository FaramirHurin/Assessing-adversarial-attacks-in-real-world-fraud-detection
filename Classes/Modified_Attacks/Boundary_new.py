from __future__ import absolute_import, division, print_function, unicode_literals

from art.attacks.evasion import BoundaryAttack

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm, trange

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format, get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)

class Boundary_new(BoundaryAttack):

    def __init__(
            self,
            estimator: "CLASSIFIER_TYPE",
            batch_size: int = 64,
            targeted: bool = True,
            delta: float = 0.01,
            epsilon: float = 0.01,
            step_adapt: float = 0.667,
            max_iter: int = 5000,
            num_trial: int = 25,
            sample_size: int = 20,
            init_size: int = 100,
            min_epsilon: float = 0.0,
            verbose: bool = True,
    ):
        super().__init__(
            estimator,
            batch_size=batch_size,
            targeted=targeted,
            delta=delta,
            epsilon=epsilon,
            step_adapt=step_adapt,
            max_iter=max_iter,
            num_trial=num_trial,
            sample_size=sample_size,
            init_size=init_size,
            min_epsilon=min_epsilon,
            verbose=verbose,
        )
        self.queries_performed = 0

    def predict_and_count(self, x):
        self.queries_performed += x.shape[0]
        return self.estimator.predict(x, batch_size=self.batch_size)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels.
        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.
        :type x_adv_init: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            y = get_labels_np_array(self.predict_and_count(x))
            #y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)

        # Get clip_min and clip_max from the classifier or infer them from data
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # Prediction from the original images
        preds = np.argmax(self.predict_and_count(x), axis=1)
        #preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        # Prediction from the initial adversarial examples if not None
        x_adv_init = kwargs.get("x_adv_init")

        if x_adv_init is not None:
            init_preds = np.argmax(self.predict_and_count(x_adv_init), axis=1)
            #init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)

        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="Boundary attack", disable=not self.verbose)):
            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=y[ind],
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            else:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        y = to_categorical(y, self.estimator.nb_classes)

        logger.info(
            "Success rate of Boundary attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _attack(
        self,
        initial_sample: np.ndarray,
        original_sample: np.ndarray,
        y_p: int,
        target: int,
        initial_delta: float,
        initial_epsilon: float,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        """
        Main function for the boundary attack.

        :param initial_sample: An initial adversarial example.
        :param original_sample: The original input.
        :param y_p: The predicted label of the original input.
        :param target: The target label.
        :param initial_delta: Initial step size for the orthogonal step.
        :param initial_epsilon: Initial step size for the step towards the target.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: an adversarial example.
        """
        # Get initialization for some variables
        x_adv = initial_sample
        self.curr_delta = initial_delta
        self.curr_epsilon = initial_epsilon

        self.curr_adv = x_adv

        # Main loop to wander around the boundary
        for _ in trange(self.max_iter, desc="Boundary attack - iterations", disable=not self.verbose):
            # Trust region method to adjust delta
            for _ in range(self.num_trial):
                potential_advs = []
                for _ in range(self.sample_size):
                    potential_adv = x_adv + self._orthogonal_perturb(self.curr_delta, x_adv, original_sample)
                    potential_adv = np.clip(potential_adv, clip_min, clip_max)
                    potential_advs.append(potential_adv)

                preds = np.argmax(
                    self.predict_and_count(np.array(potential_advs)),
                    axis=1,
                )

                '''preds = np.argmax(
                    self.estimator.predict(np.array(potential_advs), batch_size=self.batch_size),
                    axis=1,
                )
                '''

                if self.targeted:
                    satisfied = preds == target
                else:
                    satisfied = preds != y_p

                delta_ratio = np.mean(satisfied)

                if delta_ratio < 0.2:
                    self.curr_delta *= self.step_adapt
                elif delta_ratio > 0.5:
                    self.curr_delta /= self.step_adapt

                if delta_ratio > 0:
                    x_advs = np.array(potential_advs)[np.where(satisfied)[0]]
                    break
            else:  # pragma: no cover
                logger.warning("Adversarial example found but not optimal.")
                return x_adv

            # Trust region method to adjust epsilon
            for _ in range(self.num_trial):
                perturb = np.repeat(np.array([original_sample]), len(x_advs), axis=0) - x_advs
                perturb *= self.curr_epsilon
                potential_advs = x_advs + perturb
                potential_advs = np.clip(potential_advs, clip_min, clip_max)
                preds = np.argmax(
                    self.predict_and_count(potential_advs),
                    axis=1,
                )
                '''
                preds = np.argmax(
                    self.estimator.predict(potential_advs, batch_size=self.batch_size),
                    axis=1,
                )
                '''

                if self.targeted:
                    satisfied = preds == target
                else:
                    satisfied = preds != y_p

                epsilon_ratio = np.mean(satisfied)

                if epsilon_ratio < 0.2:
                    self.curr_epsilon *= self.step_adapt
                elif epsilon_ratio > 0.5:
                    self.curr_epsilon /= self.step_adapt

                if epsilon_ratio > 0:
                    x_adv = self._best_adv(original_sample, potential_advs[np.where(satisfied)[0]])
                    self.curr_adv = x_adv
                    break
            else:  # pragma: no cover
                logger.warning("Adversarial example found but not optimal.")
                return self._best_adv(original_sample, x_advs)

            if self.curr_epsilon < self.min_epsilon:
                return x_adv

        return x_adv


    def _init_sample(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        clip_min: float,
        clip_max: float,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Find initial adversarial example for the attack.

        :param x: An array with one original input to be attacked.
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :param y_p: The predicted label of x.
        :param init_pred: The predicted label of the initial image.
        :param adv_init: Initial array to act as an initial adversarial example.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: an adversarial example.
        """
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            # Attack satisfied
            if y == y_p:
                return None

            # Attack unsatisfied yet and the initial image satisfied
            if adv_init is not None and init_pred == y:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            # Attack unsatisfied yet and the initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(
                    self.predict_and_count(np.array([random_img])),
                    axis=1,
                )[0]
                '''
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]
                '''

                if random_class == y:
                    initial_sample = random_img, random_class

                    logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            # The initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(
                    self.predict_and_count(np.array([random_img])),
                    axis=1,
                )[0]
                '''
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]
                '''

                if random_class != y_p:
                    initial_sample = random_img, random_class

                    logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:  # pragma: no cover
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample



