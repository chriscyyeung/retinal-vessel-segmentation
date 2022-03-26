import tensorflow as tf


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, name="dice_loss", smooth=1e-5):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        intersect = tf.math.reduce_sum(y_pred * y_true)
        predicted_sum = tf.math.reduce_sum(y_pred * y_pred)
        gt_sum = tf.math.reduce_sum(y_true * y_true)

        loss = (2 * intersect + self.smooth) / (gt_sum + predicted_sum + self.smooth)
        loss = 1 - loss
        return loss
