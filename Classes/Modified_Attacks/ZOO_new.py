from __future__ import absolute_import, division, print_function, unicode_literals
from art.attacks.evasion import ZooAttack

import logging
from typing import Optional, Tuple, Any, TYPE_CHECKING

import numpy as np
from scipy.ndimage import zoom
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import (
    compute_success,
    get_labels_np_array,
    check_and_transform_label_format,
)
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)



class ZooAttack_new(ZooAttack):
    def __init__(
            self,
            classifier: "CLASSIFIER_TYPE",
            confidence: float = 0.0,
            targeted: bool = False,
            learning_rate: float = 1e-2,
            max_iter: int = 10,
            binary_search_steps: int = 1,
            initial_const: float = 1e-3,
            abort_early: bool = True,
            use_resize: bool = True,
            use_importance: bool = True,
            nb_parallel: int = 128,
            batch_size: int = 1,
            variable_h: float = 1e-4,
            verbose: bool = True,
    ):
        super().__init__(classifier, confidence, targeted, learning_rate, max_iter, binary_search_steps, initial_const,
                       abort_early, use_resize, use_importance, nb_parallel, batch_size, variable_h, verbose)
        self.queries_performed = 0

    def predict_and_count(self, x):
        self.queries_performed += x.shape[0]
        return self.estimator.predict(x, batch_size=self.batch_size)


    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: An array holding the adversarial examples.
        """
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        # Check that `y` is provided for targeted attacks
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            # y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            y = get_labels_np_array(self.predict_and_count(x))


        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute adversarial examples with implicit batching
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        x_adv_list = []
        for batch_id in trange(nb_batches, desc="ZOO", disable=not self.verbose):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            res = self._generate_batch(x_batch, y_batch)
            x_adv_list.append(res)
        x_adv = np.vstack(x_adv_list)

        # Apply clip
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            np.clip(x_adv, clip_min, clip_max, out=x_adv)

        # Log success rate of the ZOO attack
        logger.info(
            "Success rate of ZOO attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv


    def _loss(
        self, x: np.ndarray, x_adv: np.ndarray, target: np.ndarray, c_weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the loss function values.

        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :param target: An array with the target class (one-hot encoded).
        :param c_weight: Weight of the loss term aiming for classification as target.
        :return: A tuple holding the current logits, `L_2` distortion and overall loss.
        """
        l2dist = np.sum(np.square(x - x_adv).reshape(x_adv.shape[0], -1), axis=1)
        ratios = [1.0] + [
            int(new_size) / int(old_size) for new_size, old_size in zip(self.estimator.input_shape, x.shape[1:])
        ]
        # preds = self.estimator.predict(np.array(zoom(x_adv, zoom=ratios)), batch_size=self.batch_size)
        preds = self.predict_and_count(np.array(zoom(x_adv, zoom=ratios)))
        z_target = np.sum(preds * target, axis=1)
        z_other = np.max(
            preds * (1 - target) + (np.min(preds, axis=1) - 1)[:, np.newaxis] * target,
            axis=1,
        )

        if self.targeted:
            # If targeted, optimize for making the target class most likely
            loss = np.maximum(z_other - z_target + self.confidence, 0)
        else:
            # If untargeted, optimize for making any other class most likely
            loss = np.maximum(z_target - z_other + self.confidence, 0)

        return preds, l2dist, c_weight * loss + l2dist

