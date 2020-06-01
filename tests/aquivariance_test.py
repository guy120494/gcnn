import groupy.garray.C4_array as C4a
import tensorflow as tf
from groupy.gfunc.p4func_array import P4FuncArray
from groupy.gfunc.z2func_array import Z2FuncArray

from models.layers.GroupConv import GroupConv


class EquivarianceTest(tf.test.TestCase):

    def test_z2_c4_equivariance(self):
        layer = GroupConv(input_gruop='Z2', output_group='C4',
                          input_channels=1, output_channels=1, ksize=3)

        input_tensor = tf.random.uniform(shape=(1, 5, 5, 1))
        self.check_equivariance(input_tensor, layer, Z2FuncArray, P4FuncArray, C4a)

        output_tensor = layer(input_tensor)

        rotated_input = tf.image.rot90(input_tensor)
        rotated_output = layer(rotated_input)

        self.assertAllClose(rotated_output, tf.image.rot90(output_tensor), rtol=1e-5, atol=1e-3)

    def check_equivariance(self, im, layer, input_array, output_array, point_group):
        # Transform the image
        f = input_array(tf.transpose(im, perm=[0, 3, 1, 2]).numpy())
        g = point_group.rand()
        gf = g * f
        im1 = tf.convert_to_tensor(gf.v.transpose((0, 2, 3, 1)))

        # Compute
        yx = layer(im)
        yrx = layer(im1)

        # Transform the computed feature maps
        fmap1_garray = output_array(tf.transpose(yrx, perm=[0, 3, 1, 2]).numpy())
        r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 1))

        print(tf.math.abs(yx - r_fmap1_data).numpy().sum())
        self.assertAllClose(yx, r_fmap1_data, rtol=1e-5, atol=1e-3)

        return yx, r_fmap1_data

    def test_check_pooling_invariance(self):
        layer = GroupConv(input_gruop='Z2', output_group='C4',
                          input_channels=1, output_channels=1, ksize=3)

        input_tensor = tf.random.uniform(shape=(1, 5, 5, 1))

        layer1 = layer(input_tensor)
        output = self.plane_group_spatial_max_pooling(layer1, 'C4').numpy()
        layer1_rotated = layer(tf.image.rot90(input_tensor))
        rotated = self.plane_group_spatial_max_pooling(layer1_rotated, 'C4').numpy()

        print("Done")

    def plane_group_spatial_max_pooling(self, x, group):
        # coset max-pool
        x_shape = tf.shape(x).numpy()
        if group == 'Z2':
            x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], -1, 1])
        elif group == 'C4':
            x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], -1, 4])
            x = tf.unstack(x, axis=-1)
            x[0] = tf.image.rot90(x[0], 3)
            x[1] = tf.image.rot90(x[1], 2)
            x[2] = tf.image.rot90(x[2], 1)
            x = tf.stack(x, axis=-1)
        else:  # The group is D4
            x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], -1, 8])
        return tf.reduce_max(x, axis=[4])
