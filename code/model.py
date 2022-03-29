import os
import yaml
import h5py
import tqdm
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import dataloader
from loss import DiceLoss
from cenet import CENet


class Model:
    def __init__(self, config=None):
        self.pipeline_path = config["Pipeline"]

        # Training settings
        self.input_shape = config["TrainingSettings"]["InputShape"]
        self.epochs = config["TrainingSettings"]["Epochs"]
        self.patience = config["TrainingSettings"]["Patience"]
        self.batch_size = config["TrainingSettings"]["BatchSize"]
        self.initial_lr = config["TrainingSettings"]["Optimizer"]["InitialLearningRate"]
        self.lr_power = config["TrainingSettings"]["Optimizer"]["LearningRatePower"]
        self.weight_decay = config["TrainingSettings"]["Optimizer"]["WeightDecay"]
        self.momentum = config["TrainingSettings"]["Optimizer"]["Momentum"]

    def train(self):
        # Get images/labels and apply data augmentation
        with tf.device("/cpu:0"):
            # Load pipeline from yaml
            with open(self.pipeline_path, "r") as f:
                pipeline = yaml.load(f, Loader=yaml.FullLoader)

            # Get list of transforms
            transforms = []
            if pipeline["preprocess"]["train"] is not None:
                for transform in pipeline["preprocess"]["train"]:
                    try:
                        tfm_class = getattr(dataloader, transform["name"])(*[], **transform["variables"])
                    except KeyError:
                        tfm_class = getattr(dataloader, transform["name"])()
                    transforms.append(tfm_class)

        # Initialize dataloader, model, loss, optimizer
        generator = dataloader.DataLoader(transforms, self.batch_size, train=True)
        loss_fn = DiceLoss()
        metric = tf.keras.metrics.BinaryAccuracy()
        lr_schedule = PolyLearningRate(self.initial_lr, self.epochs * len(generator), self.lr_power)
        optimizer = tfa.optimizers.SGDW(self.weight_decay, lr_schedule, self.momentum)
        network = CENet(input_dim=self.input_shape)

        # Train step using eager execution
        @tf.function
        def train_step(img, label):
            with tf.GradientTape() as tape:
                out = network(img, training=True)
                loss = loss_fn(label, out)
            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
            return out, loss

        # Variables for early stopping
        best_epoch_loss = np.inf
        best_epoch_idx = 0
        last_improvement = 0

        # Training loop
        epoch_size = len(generator)
        for epoch in tqdm.tqdm(range(self.epochs)):
            epoch_loss = 0
            epoch_acc = 0
            for batch_idx in range(epoch_size):
                img, label = generator[batch_idx]
                out, loss = train_step(img, label)

                # Calculate average loss and accuracy
                epoch_loss += loss / epoch_size
                metric.update_state(label, out)
                epoch_acc += metric.result().numpy() / epoch_size
            print(f"{datetime.datetime.now()}: Epoch {epoch}: loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

            # Early stopping criteria
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_epoch_idx = epoch
                last_improvement = 0
            else:
                last_improvement += 1
            if last_improvement > self.patience:
                network.save_weights(f"weights/weights_{best_epoch_idx}.h5")
                break

            generator.on_epoch_end()
        else:
            network.save_weights(f"weights/weights_final.h5")

    def test(self):
        raise NotImplementedError


class PolyLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, max_iter, power):
        self.initial_lr = initial_lr
        self.max_iter = max_iter
        self.power = power

    def __call__(self, curr_iter):
        return self.initial_lr * (1 - curr_iter / self.max_iter) ** self.power
