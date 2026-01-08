import tensorflow as tf

class ParityMetric(tf.keras.Metric):
  def __init__(self, name='parity_metric', **kwargs):
    super().__init__(dtype=tf.int32, name=name, **kwargs)
    self.same_side = self.add_variable(
            shape=(),
            initializer='zeros',
            dtype=tf.int32,
            name='same_side'
        )
    self.other_side = self.add_variable(
            shape=(),
            initializer='zeros',
            dtype=tf.int32,
            name='other_side'
        )
  
  def result(self):
    return self.same_side / (self.same_side + self.other_side)

  def update_state(self, y_true, y_pred, sample_weight=None):
    true_sign = tf.keras.ops.sign(y_true)
    pred_sign = tf.keras.ops.sign(y_pred)
    sames = tf.keras.ops.equal(true_sign, pred_sign)
    diffs = tf.keras.ops.not_equal(true_sign, pred_sign)

    sames = tf.keras.ops.cast(sames, tf.int32)
    diffs = tf.keras.ops.cast(diffs, tf.int32)
    self.same_side.assign(tf.keras.ops.sum(sames))
    self.other_side.assign(tf.keras.ops.sum(diffs))

    

