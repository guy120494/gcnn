import tensorflow as tf

from models.cifar10.DenseInvariantModel import DenseInvariantModel
from models.layers.EquivariantDense import DenseMaxPooling, EquivariantDense


class InvariantDenseMaxPoolingTest(tf.test.TestCase):

    def test_two_inputs__one_is_rotation_of_the_other__return_same_result(self):
        max_pooling = DenseMaxPooling()
        input = tf.random.uniform(shape=(1, 100))
        rotated_input = tf.roll(input, shift=25, axis=1)

        result = max_pooling(input)
        result_of_rotated = max_pooling(rotated_input)

        self.assertAllEqual(result, result_of_rotated, msg="Results are not the same")


class EquivariantDenseTest(tf.test.TestCase):

    def test_two_inputs__one_is_rotation_of_the_other__return_rotated_result(self):
        equivariant_dense_layer = EquivariantDense(output_number=100)
        input = tf.random.uniform(shape=(1, 8 * 8 * 256))
        rotated_input = tf.roll(input, shift=8 * 8 * 256 // 4, axis=1)

        result = equivariant_dense_layer(input)
        result_of_rotated = equivariant_dense_layer(rotated_input)

        self.assertAllClose(tf.roll(result, shift=25, axis=0), result_of_rotated, rtol=1e-5, atol=1e-3)


class InvariantModelTest(tf.test.TestCase):
    def test_two_inputs__model_is_trimmed__one_is_rotation_of_the_other__return_same_result(self):
        inputs_shape = (1, 5, 5, 3)
        invariant_model = DenseInvariantModel()
        invariant_model.build(input_shape=inputs_shape)
        invariant_model = tf.keras.Sequential(invariant_model.layers[:-3])

        inputs = tf.random.uniform(shape=inputs_shape)
        result = invariant_model(inputs)
        # result_of_rotated = check_equivariance(inputs, invariant_model, Z2FuncArray, P4FuncArray, C4a)
        result_of_rotated = invariant_model(tf.image.rot90(inputs))

        for i in range(result.shape[1]):
            temp_result = tf.roll(result, shift=i, axis=-1)
            if tf.reduce_all(temp_result == result_of_rotated):
                print(i)
        # self.assertAllClose(result, result_of_rotated, rtol=1e-5, atol=1e-3)

    def test_two_inputs__one_is_rotation_of_the_other__return_same_result(self):
        invariant_model = DenseInvariantModel()
        invariant_model.build(input_shape=(2, 32, 32, 3))
        inputs = tf.random.uniform(shape=(2, 32, 32, 3))
        rotated_inputs = tf.image.rot90(inputs)

        result = invariant_model(inputs)
        result_of_rotated = invariant_model(rotated_inputs)

        self.assertAllEqual(tf.image.rot90(result), result_of_rotated, msg="Results are not the same")


def check_equivariance(im, model, input_array, output_array, point_group):
    # Transform the image
    f = input_array(tf.transpose(im, perm=[0, 3, 1, 2]).numpy())
    g = point_group.rand()
    gf = g * f
    im1 = tf.convert_to_tensor(gf.v.transpose((0, 2, 3, 1)))

    # Compute
    yrx = model(im1)

    # Transform the computed feature maps
    fmap1_garray = output_array(tf.transpose(yrx, perm=[0, 3, 1, 2]).numpy())
    r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 1))

    return r_fmap1_data
