import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Input, Conv2D, Conv2DTranspose, ReLU, MaxPool2D, BatchNormalization, Activation
)
from classification_models.tfkeras import Classifiers


class DACBlock(Layer):
    def __init__(self, channels):
        super().__init__()

        self.atrous1 = Conv2D(
            channels, kernel_size=3, padding="same", dilation_rate=1, kernel_initializer="he_uniform")
        self.atrous2 = Conv2D(
            channels, kernel_size=3, padding="same", dilation_rate=3, kernel_initializer="he_uniform")
        self.atrous3 = Conv2D(
            channels, kernel_size=3, padding="same", dilation_rate=5, kernel_initializer="he_uniform")
        self.conv1x1 = Conv2D(channels, kernel_size=1, kernel_initializer="he_uniform")
        self.relu = ReLU()

    def call(self, x):
        branch1 = self.relu(self.atrous1(x))
        branch2 = self.relu(self.conv1x1(self.atrous2(x)))
        branch3 = self.relu(self.conv1x1(self.atrous2(self.atrous1(x))))
        branch4 = self.relu(self.conv1x1(self.atrous3(self.atrous2(self.atrous1(x)))))
        output = x + branch1 + branch2 + branch3 + branch4
        return output


class RMPBlock(Layer):
    def __init__(self):
        super().__init__()

        self.pool1 = MaxPool2D(pool_size=2, strides=2)
        self.pool2 = MaxPool2D(pool_size=3, strides=3)
        self.pool3 = MaxPool2D(pool_size=5, strides=5)
        self.pool4 = MaxPool2D(pool_size=6, strides=6)
        self.conv = Conv2D(1, kernel_size=1, kernel_initializer="he_uniform")

    def call(self, x):
        height, width = x.shape[1:3]
        pool1_out = tf.image.resize(self.conv(self.pool1(x)), size=(height, width))
        pool2_out = tf.image.resize(self.conv(self.pool2(x)), size=(height, width))
        pool3_out = tf.image.resize(self.conv(self.pool3(x)), size=(height, width))
        pool4_out = tf.image.resize(self.conv(self.pool4(x)), size=(height, width))
        output = tf.concat([pool1_out, pool2_out, pool3_out, pool4_out, x], axis=3)
        return output


class DecoderBlock(Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = Conv2D(in_channels // 4, kernel_size=1, kernel_initializer="he_uniform")
        self.norm1 = BatchNormalization()
        self.relu1 = ReLU()

        self.deconv = Conv2DTranspose(
            in_channels // 4, kernel_size=3, strides=2,
            padding="same", output_padding=1, kernel_initializer="he_uniform"
        )
        self.norm2 = BatchNormalization()
        self.relu2 = ReLU()

        self.conv3 = Conv2D(out_channels, kernel_size=1, kernel_initializer="he_uniform")
        self.norm3 = BatchNormalization()
        self.relu3 = ReLU()

    def call(self, x, training):
        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.relu1(x)
        x = self.deconv(x)
        x = self.norm2(x, training=training)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x, training=training)
        x = self.relu3(x)
        return x


class CENet(tf.keras.Model):
    def __init__(self, input_dim=(512, 512, 3)):
        super().__init__()
        inputs = Input(input_dim)

        # ResNet backbone
        ResNet34 = Classifiers.get("resnet34")[0]
        self.resnet = ResNet34(include_top=False, input_tensor=inputs)
        for layer in self.resnet.layers:
            layer.trainable = False  # Freeze encoder layers

        # Context encoder module
        self.DAC = DACBlock(input_dim[0])
        self.RMP = RMPBlock()

        # Decoder block
        self.decoder4 = DecoderBlock(input_dim[0] + 4, input_dim[0] // 2)  # 4 extra from RMP
        self.decoder3 = DecoderBlock(input_dim[0] // 2, input_dim[0] // 4)
        self.decoder2 = DecoderBlock(input_dim[0] // 4, input_dim[0] // 8)
        self.decoder1 = DecoderBlock(input_dim[0] // 8, input_dim[0] // 8)

        # Final deconv/conv layers to get segmentation
        self.final_deconv = Conv2DTranspose(input_dim[0] // 16, kernel_size=4, strides=2, padding="same")
        self.final_relu1 = ReLU()
        self.final_conv2 = Conv2D(input_dim[0] // 16, kernel_size=3, padding="same")
        self.final_relu2 = ReLU()
        self.final_conv3 = Conv2D(1, kernel_size=3, padding="same")
        self.sigmoid = Activation("sigmoid")

    def call(self, x, training):
        e = self.resnet(x)
        # Skip connections
        skip1 = self.resnet.get_layer("stage2_unit1_relu1").output
        skip2 = self.resnet.get_layer("stage3_unit1_relu1").output
        skip3 = self.resnet.get_layer("stage4_unit1_relu1").output

        bridge = self.DAC(e)
        bridge = self.RMP(bridge)

        d4 = self.decoder4(bridge, training=training) + skip3
        d3 = self.decoder3(d4, training=training) + skip2
        d2 = self.decoder2(d3, training=training) + skip1
        d1 = self.decoder1(d2, training=training)

        out = self.final_deconv(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        out = self.sigmoid(out)

        return out


if __name__ == '__main__':
    from dataloader import DataLoader
    dataloader = DataLoader(batch_size=8)
    test_x, test_y = dataloader[0]
    model = CENet()
    output = model(test_x)
    model.summary()
