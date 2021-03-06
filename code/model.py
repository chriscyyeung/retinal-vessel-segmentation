import yaml
import h5py
import tqdm
import datetime
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import dataloader
from metric import accuracy, calculate_eval_metric, calculate_auc_test
from loss import DiceLoss
from cenet import CENet


class Model:
    def __init__(self, config=None):
        self.pipeline_path = config["Pipeline"]

        # Training settings
        self.input_shape = config["TrainingSettings"]["InputShape"]
        self.epochs = config["TrainingSettings"]["Epochs"]
        self.patience = config["TrainingSettings"]["Patience"]
        self.lr_update_patience = config["TrainingSettings"]["UpdateLRPatience"]
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
        # loss_fn = DiceLoss()
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        metric = tf.keras.metrics.BinaryAccuracy()
        # lr_schedule = PolyLearningRate(self.initial_lr, self.epochs * len(generator), self.lr_power)
        # optimizer = tfa.optimizers.SGDW(self.weight_decay, lr_schedule, self.momentum)
        optimizer = tfa.optimizers.AdamW(self.weight_decay, self.initial_lr)
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
            print(f"{datetime.datetime.now()}: Epoch {epoch + 1}: loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

            # Early stopping criteria
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                last_improvement = 0
                network.save_weights("weights/weights_cenet_adam.h5")
            else:
                last_improvement += 1
            if last_improvement > self.patience:
                print(f"Early stop at {epoch = }.")
                break
            if last_improvement > self.lr_update_patience:
                old_lr = optimizer.lr.read_value()
                if old_lr < 5e-7:
                    break
                network.load_weights("weights/weights_cenet_adam.h5")
                new_lr = old_lr / 2.0
                optimizer.lr.assign(new_lr)
                print(f"Updating learning rate: {new_lr}")

            generator.on_epoch_end()

    def test(self, saved_weight_path=None):
        network = CENet(input_dim=self.input_shape)
        arr = np.zeros((1, 512, 512, 3))
        network(arr)
        if saved_weight_path:
            network.load_weights(saved_weight_path)
        generator = dataloader.DataLoader()
        # total_metric = 0.0
        total_auc = []
        total_acc = []
        total_sen = []
        for i in range(len(generator)):
            img, label = generator[i]
            out = network(img).numpy()
            out[out > 0.5] = 255
            out[out <= 0.5] = 0
            label = label * 255.
            out_copy = np.copy(out)
            # cv2.imwrite(f"results/{i}_test_mask.png", out[0])
            # single_metric = calculate_eval_metric(out[0], label[0])
            auc = calculate_auc_test(out_copy[0], label[0])
            out[out == 255] = 1
            label[label == 255] = 1
            acc, sen = accuracy(out[0, ..., 0], label[0, ..., 0])
            total_auc.append(auc)
            total_acc.append(acc)
            total_sen.append(sen)
            print(i, acc, sen, auc)
            # print(f"{i}: {single_metric}")
            # total_metric += np.asarray(single_metric)
        # avg_metric = total_metric / len(generator)
        # print(avg_metric)
        print(np.mean(total_acc), np.std(total_acc))
        print(np.mean(total_sen), np.std(total_sen))
        print(np.mean(total_auc), np.std(total_auc))



class PolyLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, max_iter, power):
        self.initial_lr = initial_lr
        self.max_iter = max_iter
        self.power = power

    def __call__(self, curr_iter):
        return self.initial_lr * (1 - curr_iter / self.max_iter) ** self.power
