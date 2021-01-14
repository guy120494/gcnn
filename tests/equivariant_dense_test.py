import tensorflow as tf

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
