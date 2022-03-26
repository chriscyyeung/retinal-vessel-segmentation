import os
import math
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, transforms=None, batch_size=1, train=False, shuffle=True):
        self.input_dim = (512, 512)
        self.transforms = transforms
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_path = "data/training" if train else "data/test"
        self.images, self.labels = self.get_dataset_names()

    def get_dataset_names(self):
        # Get lists of images and ground truth labels
        images = []
        labels = []
        image_root = os.path.join(self.data_path, "images")
        label_root = os.path.join(self.data_path, "1st_manual")

        for image_name in os.listdir(image_root):
            label_path = image_name.split("_")[0] + "_manual1.gif"
            images.append(os.path.join(image_root, image_name))
            labels.append(os.path.join(label_root, label_path))

        return images, labels

    def input_parser(self, img_path, label_path):
        # Read images
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.input_dim)
        label = np.array(Image.open(label_path))
        label = cv2.resize(label, self.input_dim)
        sample = {"image": img, "label": label}

        # Apply data augmentation methods
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        # Normalize pixel values and convert label to binary
        img = sample["image"] / 255.
        label = np.expand_dims(sample["label"], axis=2) / 255.
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        return img, label

    def on_epoch_end(self):
        # Shuffle dataset
        if self.shuffle:
            rng = np.random.default_rng()
            p = rng.permutation(len(self.images))
            self.images = self.images[p]
            self.labels = self.labels[p]

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        # Get names of images and labels of batch
        X_batch_names = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch_names = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Create empty arrays for images in the batch
        X_batch = np.zeros((0, *self.input_dim, 3))  # 3 color channels
        y_batch = np.zeros((0, *self.input_dim, 1))  # only one channel

        # Add images and labels to arrays
        for i in range(self.batch_size):
            img, label = self.input_parser(X_batch_names[i], y_batch_names[i])
            X_batch = np.concatenate((X_batch, np.expand_dims(img, axis=0)))
            y_batch = np.concatenate((y_batch, np.expand_dims(label, axis=0)))

        X_batch = X_batch.astype(np.float32)
        y_batch = (y_batch * 255).astype(np.uint8)
        return X_batch, y_batch


class ColorJitter:
    """Randomly adjusts the hue, saturation, and value of an image."""
    def __init__(self, hue_limit=30, sat_limit=5, val_limit=15):
        self.name = "ColorJitter"
        self.hue_limit = hue_limit
        self.sat_limit = sat_limit
        self.val_limit = val_limit
        self.p = 0.5

    def __call__(self, sample):
        rng = np.random.default_rng()
        img = sample["image"]

        if rng.random() < self.p:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV
            h, s, v = cv2.split(img)
            # Apply hue shift
            hue_shift = rng.integers(-self.hue_limit, self.hue_limit + 1)
            hue_shift = np.uint8(hue_shift)
            h += hue_shift
            # Apply saturation shift
            sat_shift = rng.uniform(-self.sat_limit, self.sat_limit)
            s += cv2.add(s, sat_shift)
            # Apply value shift
            val_shift = rng.uniform(-self.val_limit, self.val_limit)
            v = cv2.add(v, val_shift)
            # Merge changes and convert back to BGR
            img = cv2.merge((h, s, v))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        return {"image": img, "label": sample["label"]}


class RandomScale:
    """Randomly scales an image by a scaling factor."""
    def __init__(self, scaling_limit=0.1):
        self.name = "RandomScale"
        self.scaling_limit = scaling_limit

    def __call__(self, sample):
        rng = np.random.default_rng()

        # Randomly determine scaling factor
        scaling_factor = rng.uniform(1 - self.scaling_limit, 1 + self.scaling_limit)
        interpolation = cv2.INTER_AREA if scaling_factor < 1 else cv2.INTER_LINEAR

        # Generate transformation matrix
        img, label = sample["image"], sample["label"]
        height, width = img.shape[:2]
        box0 = np.array(
            [[0, 0],
             [width, 0],
             [width, height],
             [0, height]]
        )
        box1 = box0 * scaling_factor

        # Apply transformation to image and label
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = cv2.warpPerspective(img, mat, (width, height), flags=interpolation)
        label = cv2.warpPerspective(label, mat, (width, height), flags=interpolation)

        return {"image": img, "label": label}


class RandomShift:
    """Randomly shifts an image by a proportion of the image dimensions."""
    def __init__(self, shift_factor=0.1):
        self.name = "RandomShift"
        self.shift_factor = shift_factor

    def __call__(self, sample):
        rng = np.random.default_rng()

        # Randomly get distance to shift image in x and y
        img, label = sample["image"], sample["label"]
        height, width = img.shape[:2]
        max_dx = self.shift_factor * width
        max_dy = self.shift_factor * height
        dx = round(rng.uniform(-max_dx, max_dx))
        dy = round(rng.uniform(-max_dy, max_dy))

        # Generate transformation matrix
        box0 = np.array(
            [[0, 0],
             [width, 0],
             [width, height],
             [0, height]]
        )
        box1 = box0 - np.array([width / 2, height / 2])
        box1 += np.array([width / 2 + dx, height / 2 + dy])

        # Apply transformation to image and label
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR)
        label = cv2.warpPerspective(label, mat, (width, height), flags=cv2.INTER_LINEAR)

        return {"image": img, "label": label}


class RandomFlip:
    """Randomly flips an image horizontally and vertically."""
    def __init__(self):
        self.name = "RandomFlip"
        self.p = 0.5

    def __call__(self, sample):
        rng = np.random.default_rng()
        img, label = sample["image"], sample["label"]

        # Horizontal flip
        if rng.random() < self.p:
            img = np.flip(img, 1)
            label = np.flip(label, 1)

        # Vertical flip
        if rng.random() < self.p:
            img = np.flip(img, 0)
            label = np.flip(label, 0)

        return {"image": img, "label": label}


class RandomRotation:
    """Randomly rotates an image 0, 90, 180, or 270 degrees."""
    def __init__(self):
        self.name = "RandomRotation"

    def __call__(self, sample):
        rng = np.random.default_rng()
        img, label = sample["image"], sample["label"]

        num_rot = rng.integers(4)  # Number of 90 degree rotations
        img = np.rot90(img, num_rot)
        label = np.rot90(label, num_rot)

        return {"image": img, "label": label}


if __name__ == '__main__':
    import yaml

    train_dataloader = DataLoader(batch_size=8)
    img, label = train_dataloader[0]
    print(f"image type: {img.dtype}, label type: {label.dtype}")
    print(f"train image shape: {img.shape}, train label shape: {label.shape}")

    test_dataloader = DataLoader(train=False)
    test_img, test_label = test_dataloader[0]
    print(f"test image shape: {test_img.shape}, test label shape: {test_label.shape}")

    # Get list of train transforms
    with open("config/pipeline.yaml", "r") as f:
        pipeline = yaml.load(f, Loader=yaml.FullLoader)
    transforms = []
    if pipeline["preprocess"]["train"] is not None:
        for transform in pipeline["preprocess"]["train"]:
            try:
                tfm_class = locals()[transform["name"]](*[], **transform["variables"])
            except KeyError:
                tfm_class = locals()[transform["name"]]()
            transforms.append(tfm_class)

    # Test data augmentation classes
    sample = {"image": img[0], "label": label[0]}
    print(f"image shape: {img[0].shape}, label shape: {label[0].shape}")
    cv2.imshow("original image", sample["image"])
    cv2.waitKey(1000)
    cv2.imshow("original label", sample["label"])
    cv2.waitKey(1000)
    for transform in transforms:
        transformed_sample = transform(sample)
        new_img, new_label = transformed_sample["image"], transformed_sample["label"]
        cv2.imshow(transform.name, new_img)
        cv2.waitKey(1000)
        cv2.imshow(transform.name, new_label)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Test color jitter (only works with uint8 images so above code doesn't work)
    transform_dataloader = DataLoader(transforms=transforms, batch_size=8)
    transform_img, transform_label = transform_dataloader[0]
    for i in range(transform_img.shape[0]):
        cv2.imshow(f"image {i + 1}", transform_img[i])
        cv2.waitKey(1000)
