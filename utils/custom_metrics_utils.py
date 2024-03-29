# from https://towardsdatascience.com/f-beta-score-in-keras-part-ii-15f91f07c9a4
import tensorflow as tf
from tensorflow.keras.metrics import Metric


class StatefullMultiClassFBeta(Metric):

    """
    Custom Keras Fbeta metric used to calculate F beta during modeling
    """

    def __init__(
        self,
        name="state_full_binary_fbeta",
        beta=1,
        n_class=2,
        average="macro",
        epsilon=1e-7,
        **kwargs
    ):
        """
        Custom Keras Fbeta metric used to calculate Fbeta during modeling.

        Parameters
        ----------
            name (str, optional): The name of the metric. Defaults to
            "state_full_binary_fbeta".
            beta (int, optional): The beta value used in Fbeta formula.
            Defaults to 1.
            n_class (int, optional): The number of classes in the data.
            Defaults to 2.
            average (str, optional): The type of average to use in Fbeta
            calculation. Can be either "macro", "weighted", or "raw".
            Defaults to "macro".
            epsilon (float, optional): The small value used to avoid division
            by zero. Defaults to 1e-7.
            kwargs: Additional keyword arguments.

        Returns
        -------
            tf.reduce_mean or tf.reduce_sum: The average or weighted average of
            Fbeta score depending on the `average` parameter.
        """
        # initializing an object of the super class
        super(StatefullMultiClassFBeta, self).__init__(name=name, **kwargs)

        # initializing state variables
        # initializing true positives
        self.tp = self.add_weight(name="tp", shape=(n_class,), initializer="zeros")

        # initializing actual positives
        self.actual_positives = self.add_weight(
            name="ap", shape=(n_class,), initializer="zeros"
        )
        # initializing predicted positives
        self.predicted_positives = self.add_weight(
            name="pp", shape=(n_class,), initializer="zeros"
        )

        # initializing other atrributes that won't be
        # changed for every object of this class
        self.beta_squared = beta**2
        self.n_class = n_class
        self.average = average
        self.epsilon = epsilon

    def update_state(
        self, ytrue: tf.Tensor, ypred: tf.Tensor, sample_weight: tf.Tensor = None
    ) -> None:
        """
        Updates the state variables during training.

        Parameters
        ----------
            ytrue (tf.Tensor): The ground truth labels.
            ypred (tf.Tensor): The predicted labels.
            sample_weight (tf.Tensor, optional): The weight of each sample. Defaults to None.

        Returns
        -------
            None
        """
        # casting ytrue and ypred as float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)

        # finding the maximum probability in ypred
        max_prob = tf.reduce_max(ypred, axis=-1, keepdims=True)

        # making ypred one hot encoded so class with the maximum probability
        # will be encoded as 1 while others as 0
        ypred = tf.cast(tf.equal(ypred, max_prob), tf.float32)

        # updating true positives atrribute
        self.tp.assign_add(tf.reduce_sum(ytrue * ypred, axis=0))

        # updating predicted positives atrribute
        self.predicted_positives.assign_add(tf.reduce_sum(ypred, axis=0))

        # updating actual positives atrribute
        self.actual_positives.assign_add(tf.reduce_sum(ytrue, axis=0))

    def result(self):
        """
        Calculates and returns F beta score

        Returns
        -------
            tf.reduce_mean or tf.reduce_sum: The average or weighted average of
            Fbeta score depending on the `average` parameter.
        """
        self.precision = self.tp / (self.predicted_positives + self.epsilon)
        self.recall = self.tp / (self.actual_positives + self.epsilon)

        # calculating fbeta score
        self.fb = (
            (1 + self.beta_squared)
            * self.precision
            * self.recall
            / (self.beta_squared * self.precision + self.recall + self.epsilon)
        )

        if self.average == "weighted":
            return tf.reduce_sum(
                self.fb * self.actual_positives / tf.reduce_sum(self.actual_positives)
            )

        elif self.average == "raw":
            return self.fb

        return tf.reduce_mean(self.fb)

    def reset_states(self):
        """
        Resets all states

        Returns
        -------
            None
        """
        self.tp.assign(tf.zeros(self.n_class))
        self.predicted_positives.assign(tf.zeros(self.n_class))
        self.actual_positives.assign(tf.zeros(self.n_class))
