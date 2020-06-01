from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
from tensorflow.keras import layers


class GroupConv(layers.Layer):
    def __init__(self, input_gruop, output_group, input_channels, output_channels, ksize):
        super(GroupConv, self).__init__()
        self.ksize = ksize
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.input_group = input_gruop
        self.output_group = output_group
        self.gconv_indices, self.gconv_shape_info, self.w_shape, self.w = None, None, None, None

    def build(self, input_shape):
        self.gconv_indices, self.gconv_shape_info, self.w_shape = gconv2d_util(
            h_input=self.input_group, h_output=self.output_group, in_channels=self.input_channels,
            out_channels=self.output_channels, ksize=self.ksize)

        self.w = self.add_weight(shape=self.w_shape, initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        x = gconv2d(input=inputs, filter=self.w, strides=[1, 1, 1, 1], padding='SAME',
                    gconv_indices=self.gconv_indices, gconv_shape_info=self.gconv_shape_info)
        return x
