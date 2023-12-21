import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 'gradio'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'tensorflow-addons'])

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from transformers.tf_utils import shape_list
import numpy as np
import cv2
from PIL import Image
import gradio as gr

class SelectiveKernelFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_channels = max(int(channels / 8), 4)

        self.average_pooling = tfa.layers.AdaptiveAveragePooling2D(output_size=1)

        self.conv_channel_downscale = tf.keras.layers.Conv2D(
            self.hidden_channels, kernel_size=1, padding="same"
        )
        self.conv_attention_1 = tf.keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, padding="same"
        )
        self.conv_attention_2 = tf.keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, padding="same"
        )

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.sorftmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs, *args, **kwargs):
        combined_input_features = inputs[0] + inputs[1]
        channel_wise_statistics = self.average_pooling(combined_input_features)
        downscaled_channel_wise_statistics = self.conv_channel_downscale(
            channel_wise_statistics
        )
        attention_vector_1 = self.sorftmax(
            self.conv_attention_1(downscaled_channel_wise_statistics)
        )
        attention_vector_2 = self.sorftmax(
            self.conv_attention_2(downscaled_channel_wise_statistics)
        )
        selected_features = (
            inputs[0] * attention_vector_1 + inputs[1] * attention_vector_2
        )
        return selected_features


class ContextBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask_conv = tf.keras.layers.Conv2D(
            1, kernel_size=1, padding="same"
        )

        self.channel_add_conv_1 = tf.keras.layers.Conv2D(
            channels, kernel_size=1, padding="same"
        )
        self.channel_add_conv_2 = tf.keras.layers.Conv2D(
            channels, kernel_size=1, padding="same"
        )

        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def modeling(self, inputs):
        _, height, width, channels = shape_list(inputs)
        reshaped_inputs = tf.expand_dims(
            tf.reshape(inputs, (-1, channels, height * width)), axis=1
        )

        context_mask = self.mask_conv(inputs)
        context_mask = tf.reshape(context_mask, (-1, height * width, 1))
        context_mask = self.softmax(context_mask)
        context_mask = tf.expand_dims(context_mask, axis=1)

        context = tf.reshape(
            tf.matmul(reshaped_inputs, context_mask), (-1, 1, 1, channels)
        )
        return context

    def call(self, inputs, *args, **kwargs):
        context = self.modeling(inputs)
        channel_add_term = self.channel_add_conv_1(context)
        channel_add_term = self.leaky_relu(channel_add_term)
        channel_add_term = self.channel_add_conv_2(channel_add_term)
        return inputs + channel_add_term


class ResidualContextBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, groups: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(
            channels, kernel_size=3, padding="same", groups=groups
        )
        self.conv_2 = tf.keras.layers.Conv2D(
            channels, kernel_size=3, padding="same", groups=groups
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.context_block = ContextBlock(channels=channels)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        x = self.context_block(x)
        x = self.leaky_relu(x)
        x = x + inputs
        return x

class DownBlock(tf.keras.layers.Layer):
    def __init__(
        self, channels: int, channel_factor: float, *args, **kwargs
    ):
        super(DownBlock, self).__init__(*args, **kwargs)
        self.average_pool = tf.keras.layers.AveragePooling2D(
            pool_size=2, strides=2
        )
        self.conv = tf.keras.layers.Conv2D(
            int(channels * channel_factor),
            kernel_size=1,
            strides=1,
            padding="same"
        )

    def call(self, inputs, *args, **kwargs):
        return self.conv(self.average_pool(inputs))


class DownSampleBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        scale_factor: int,
        channel_factor: float,
        *args,
        **kwargs
    ):
        super(DownSampleBlock, self).__init__(*args, **kwargs)
        self.layers = []
        for _ in range(int(np.log2(scale_factor))):
            self.layers.append(DownBlock(channels, channel_factor))
            channels = int(channels * channel_factor)

    def call(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x

class UpBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, channel_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = tf.keras.layers.Conv2D(
            int(channels // channel_factor), kernel_size=1, strides=1, padding="same"
        )
        self.upsample = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")

    def call(self, inputs, *args, **kwargs):
        return self.upsample(self.conv(inputs))


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(
        self, channels: int, scale_factor: int, channel_factor: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layers = []
        for _ in range(int(np.log2(scale_factor))):
            self.layers.append(UpBlock(channels, channel_factor))
            channels = int(channels // channel_factor)

    def call(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiScaleResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        channel_factor: float,
        groups: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Residual Context Blocks
        self.rcb_top = ResidualContextBlock(
            int(channels * channel_factor**0), groups=groups
        )
        self.rcb_middle = ResidualContextBlock(
            int(channels * channel_factor**1), groups=groups
        )
        self.rcb_bottom = ResidualContextBlock(
            int(channels * channel_factor**2), groups=groups
        )

        # Downsample Blocks
        self.down_2 = DownSampleBlock(
            channels=int((channel_factor**0) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.down_4_1 = DownSampleBlock(
            channels=int((channel_factor ** 0) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.down_4_2 = DownSampleBlock(
            channels=int((channel_factor ** 1) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )

        # UpSample Blocks
        self.up21_1 = UpSampleBlock(
            channels=int((channel_factor ** 1) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.up21_2 = UpSampleBlock(
            channels=int((channel_factor ** 1) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.up32_1 = UpSampleBlock(
            channels=int((channel_factor ** 2) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.up32_2 = UpSampleBlock(
            channels=int((channel_factor ** 2) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )

        # SKFF Blocks
        self.skff_top = SelectiveKernelFeatureFusion(
            channels=int(channels * channel_factor**0)
        )
        self.skff_middle = SelectiveKernelFeatureFusion(
            channels=int(channels * channel_factor**1)
        )

        # Convolution
        self.conv_out = tf.keras.layers.Conv2D(
            channels, kernel_size=1, padding="same"
        )

    def call(self, inputs, *args, **kwargs):
        x_top = inputs
        x_middle = self.down_2(x_top)
        x_bottom = self.down_4_2(self.down_4_1(x_top))

        x_top = self.rcb_top(x_top)
        x_middle = self.rcb_middle(x_middle)
        x_bottom = self.rcb_bottom(x_bottom)

        x_middle = self.skff_middle([x_middle, self.up32_1(x_bottom)])
        x_top = self.skff_top([x_top, self.up21_1(x_middle)])

        x_top = self.rcb_top(x_top)
        x_middle = self.rcb_middle(x_middle)
        x_bottom = self.rcb_bottom(x_bottom)

        x_middle = self.skff_middle([x_middle, self.up32_2(x_bottom)])
        x_top = self.skff_top([x_top, self.up21_2(x_middle)])

        output = self.conv_out(x_top)
        output = output + inputs

        return output

class RecursiveResidualGroup(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        num_mrb_blocks: int,
        channel_factor: float,
        groups: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layers = [
            MultiScaleResidualBlock(channels, channel_factor, groups)
            for _ in range(num_mrb_blocks)
        ]
        self.layers.append(
            tf.keras.layers.Conv2D(
                channels, kernel_size=3, strides=1, padding="same"
            )
        )

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        for layer in self.layers:
            residual = layer(residual)
        residual = residual + inputs
        return residual

@keras.saving.register_keras_serializable()
class MirNetv2(tf.keras.Model):
    def __init__(
        self,
        channels: int,
        channel_factor: float,
        num_mrb_blocks: int,
        add_residual_connection: bool,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.add_residual_connection = add_residual_connection

        self.conv_in = tf.keras.layers.Conv2D(
            channels, kernel_size=3, padding="same"
        )

        self.rrg_block_1 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=1
        )
        self.rrg_block_2 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=2
        )
        self.rrg_block_3 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=4
        )
        self.rrg_block_4 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=4
        )

        self.conv_out = tf.keras.layers.Conv2D(
            3, kernel_size=3, padding="same"
        )

    def call(self, inputs, training=None, mask=None):
        shallow_features = self.conv_in(inputs)
        deep_features = self.rrg_block_1(shallow_features)
        deep_features = self.rrg_block_2(deep_features)
        deep_features = self.rrg_block_3(deep_features)
        deep_features = self.rrg_block_4(deep_features)
        output = self.conv_out(deep_features)
        output = output + inputs if self.add_residual_connection else output
        return output


@keras.saving.register_keras_serializable()
class MirNetv2SuperResVer1(tf.keras.Model):
    def __init__(
        self,
        channels: int,
        channel_factor: float,
        num_mrb_blocks: int,
        add_residual_connection: bool,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.add_residual_connection = add_residual_connection

        self.upsample = tf.keras.layers.UpSampling2D(
            size=4, interpolation="bilinear"
        )

        self.conv_in = tf.keras.layers.Conv2D(
            channels, kernel_size=3, padding="same"
        )

        self.rrg_block_1 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=1
        )
        self.rrg_block_2 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=2
        )
        self.rrg_block_3 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=4
        )
        self.rrg_block_4 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=4
        )

        self.conv_out = tf.keras.layers.Conv2D(
            3, kernel_size=3, padding="same"
        )
    def call(self, inputs, training=None, mask=None):
        inputs = self.upsample(inputs)
        shallow_features = self.conv_in(inputs)
        deep_features = self.rrg_block_1(shallow_features)
        deep_features = self.rrg_block_2(deep_features)
        deep_features = self.rrg_block_3(deep_features)
        deep_features = self.rrg_block_4(deep_features)
        output = self.conv_out(deep_features)
        output = output + inputs if self.add_residual_connection else output
        return output


class UpScale2X(tf.keras.layers.Layer):
  def __init__(self, channels: int, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.upSampling2D = keras.layers.UpSampling2D(
        size=2, interpolation='bilinear'
    )

    self.conv2D = keras.layers.Conv2D(
        channels, kernel_size=3, strides=1, padding='same'
    )

    self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

  def call(self, input, *args, **kwargs):
    x = self.upSampling2D(input)
    x = self.conv2D(x)
    x = self.leaky_relu(x)
    return x


class HRConv(tf.keras.layers.Layer):
  def __init__(self, channels: int, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.Conv2D = keras.layers.Conv2D(
        channels, kernel_size=3, strides=1, padding='same'
    )

    self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

  def call(self, input, *args, **kwargs):
    x = self.Conv2D(input)
    x = self.leaky_relu(x)
    return x


#use ESRGAN UpSampling2D structure
@keras.saving.register_keras_serializable()
class MirNetv2SuperResVer2(tf.keras.Model):
    def __init__(
        self,
        channels: int,
        channel_factor: float,
        num_mrb_blocks: int,
        add_residual_connection: bool,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.add_residual_connection = add_residual_connection

        self.conv_in = tf.keras.layers.Conv2D(
            channels, kernel_size=3, padding="same"
        )

        self.rrg_block_1 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=1
        )
        self.rrg_block_2 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=2
        )
        self.rrg_block_3 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=4
        )
        self.rrg_block_4 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=4
        )

        self.upScale2X_1 = UpScale2X(channels)
        self.upScale2X_2 = UpScale2X(channels)

        self.HRConv = HRConv(channels)

        self.conv_out = tf.keras.layers.Conv2D(
            3, kernel_size=3, padding="same"
        )
    def call(self, inputs, training=None, mask=None):
        shallow_features = self.conv_in(inputs)

        deep_features = self.rrg_block_1(shallow_features)
        deep_features = self.rrg_block_2(deep_features)
        deep_features = self.rrg_block_3(deep_features)
        deep_features = self.rrg_block_4(deep_features)

        upScale2X = self.upScale2X_1(deep_features)
        upScale4X = self.upScale2X_2(upScale2X)

        output = self.conv_out(self.HRConv(upScale4X))
        output = output + inputs if self.add_residual_connection else output
        return output
    #def get_config(self):
    #    return {"channels": self.channels, "channel_factor": self.channel_factor, "num_mrb_blocks": self.num_mrb_blocks, "add_residual_connection": self.add_residual_connection}