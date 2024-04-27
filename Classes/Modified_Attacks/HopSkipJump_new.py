from __future__ import absolute_import, division, print_function, unicode_literals

from art.attacks.evasion import HopSkipJump

import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format, get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)

class HopSkipJump_new(HopSkipJump):
    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        batch_size: int = 64,
        targeted: bool = False,
        norm: Union[int, float, str] = 2,
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        verbose: bool = True,
    ):
        super().__init__(
            classifier,
            batch_size=batch_size,
            targeted=targeted,
            norm=norm,
            max_iter=max_iter,
            max_eval=max_eval,
            init_eval=init_eval,
            init_size=init_size,
            verbose=verbose,)
        self.queries_performed = 0


    def predict_and_count(self, x):
        self.queries_performed += x.shape[0]
        return self.estimator.predict(x, batch_size=self.batch_size)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,).
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.
        :type x_adv_init: `np.ndarray`
        :param resume: Allow users to continue their previous attack.
        :type resume: `bool`
        :return: An array holding the adversarial examples.
        """
        mask = kwargs.get("mask")

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            y = get_labels_np_array(self.predict_and_count(x))
            #y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Check whether users need a stateful attack
        resume = kwargs.get("resume")

        if resume is not None and resume:
            start = self.curr_iter
        else:
            start = 0

        # Check the mask
        if mask is not None:
            if len(mask.shape) == len(x.shape):
                mask = mask.astype(ART_NUMPY_DTYPE)
            else:
                mask = np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])
        else:
            mask = np.array([None] * x.shape[0])

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
            # Add mask param to the x_adv_init
            for i in range(x.shape[0]):
                if mask[i] is not None:
                    x_adv_init[i] = x_adv_init[i] * mask[i] + x[i] * (1 - mask[i])

            # Do prediction on the init
            init_preds = np.argmax(self.predict_and_count(x_adv_init), axis=1)
            # init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)

        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)

        y = np.argmax(y, axis=1)

        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="HopSkipJump", disable=not self.verbose)):
            self.curr_iter = start

            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=y[ind],  # type: ignore
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    mask=mask[ind],
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
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        y = to_categorical(y, self.estimator.nb_classes)  # type: ignore

        logger.info(
            "Success rate of HopSkipJump attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _init_sample(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, int]]]:
        """
        Find initial adversarial example for the attack.

        :param x: An array with 1 original input to be attacked.
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :param y_p: The predicted label of x.
        :param init_pred: The predicted label of the initial image.
        :param adv_init: Initial array to act as an initial adversarial example.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: An adversarial example.
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

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)
                random_class = np.argmax(
                    self.predict_and_count(np.array([random_img])),
                    axis=1,
                )[0]
                '''random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]
                '''

                if random_class == y:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, random_class

                    logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), y_p

            # The initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                random_class = np.argmax(
                    self.predict_and_count(np.array([random_img])),
                    axis=1,
                )[0]

                '''random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]
                '''

                if random_class != y_p:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y_p,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, y_p

                    logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample


    def _adversarial_satisfactory(
        self, samples: np.ndarray, target: int, clip_min: float, clip_max: float
    ) -> np.ndarray:
        """
        Check whether an image is adversarial.

        :param samples: A batch of examples.
        :param target: The target label.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: An array of 0/1.
        """
        samples = np.clip(samples, clip_min, clip_max)
        preds = np.argmax(self.predict_and_count(samples), axis=1)
        #preds = np.argmax(self.estimator.predict(samples, batch_size=self.batch_size), axis=1)

        if self.targeted:
            result = preds == target
        else:
            result = preds != target

        return result