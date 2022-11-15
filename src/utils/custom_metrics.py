#https://towardsdatascience.com/f-beta-score-in-keras-part-ii-15f91f07c9a4
import tensorflow as tf


class StatefullMultiClassFBeta(Metric):
  def __init__(self, name='state_full_binary_fbeta', beta=1, n_class=2, average='macro', epsilon=1e-7, **kwargs):
    # initializing an object of the super class
    super(StatefullMultiClassFBeta, self).__init__(name=name, **kwargs)

    # initializing state variables
    self.tp = self.add_weight(name='tp', shape=(n_class,), initializer='zeros')     # initializing true positives
    self.actual_positives = self.add_weight(name='ap', shape=(n_class,), initializer='zeros') # initializing actual positives
    self.predicted_positives = self.add_weight(name='pp', shape=(n_class,), initializer='zeros') # initializing predicted positives

    # initializing other atrributes that wouldn't be changed for every object of this class
    self.beta_squared = beta**2
    self.n_class = n_class
    self.average = average
    self.epsilon = epsilon

  def update_state(self, ytrue, ypred, sample_weight=None):
    # casting ytrue and ypred as float dtype
    ytrue = tf.cast(ytrue, tf.float32)
    ypred = tf.cast(ypred, tf.float32)

    # finding the maximum probability in ypred
    max_prob = tf.reduce_max(ypred, axis=-1, keepdims=True)

    # making ypred one hot encoded such that the class with the maximum probability as encoded as 1 while others as 0
    ypred = tf.cast(tf.equal(ypred, max_prob), tf.float32)
        
    self.tp.assign_add(tf.reduce_sum(ytrue*ypred, axis=0)) # updating true positives atrribute
    self.predicted_positives.assign_add(tf.reduce_sum(ypred, axis=0)) # updating predicted positives atrribute
    self.actual_positives.assign_add(tf.reduce_sum(ytrue, axis=0)) # updating actual positives atrribute

  def result(self):
    self.precision = self.tp/(self.predicted_positives+self.epsilon) # calculates precision
    self.recall = self.tp/(self.actual_positives+self.epsilon) # calculates recall

    # calculating fbeta score
    self.fb = (1+self.beta_squared)*self.precision*self.recall / (self.beta_squared*self.precision + self.recall + self.epsilon)

    if self.average == 'weighted':
      return tf.reduce_sum(self.fb*self.actual_positives / tf.reduce_sum(self.actual_positives))
    
    elif self.average == 'raw':
      return self.fb

    return tf.reduce_mean(self.fb)

  def reset_states(self):
    self.tp.assign(tf.zeros(self.n_class)) # resets true positives to zero
    self.predicted_positives.assign(tf.zeros(self.n_class)) # resets predicted positives to zero
    self.actual_positives.assign(tf.zeros(self.n_class)) # resets actual positives to zero