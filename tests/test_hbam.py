import unittest
import numpy as np

from .. import hbam


class TestHBAMMethods(unittest.TestCase):

    def test_arr2int_simple(self):
        input = np.array([0, 0, 0, 1])
        output = hbam.arr2int(input)
        self.assertEqual(output, 1)

    def test_arr2int_nonbinary(self):
        input = np.array([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            hbam.arr2int(input)

    def test_arr2int_input_too_large(self):
        input = np.random.rand(100)
        with self.assertRaises(AssertionError):
            hbam.arr2int(input)

    def test_binarize(self):
        input = np.array([3, 2, 1, 0, 3, 0, 1, 0])
        output = hbam.binarize((input)).tolist()
        self.assertListEqual(output, [1, 1, 1, 0, 1, 0, 1, 0])

    def test_unbinarize(self):
        input = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        output = hbam.unbinarize(input, signature_size=2).tolist()
        self.assertListEqual(output, [0, 1, 2, 3])

    def test_length_of_embedding(self):
        input = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        output = hbam.seq2hbseq(input)

        self.assertEqual(len(output), hbam.EMBEDDING_SIZE)

    def test_weighted_cosine(self):
        x = np.array([1, 1, 1])
        y = np.array([1, 2, 0])
        w = np.array([1, 0.5, 0.1])

        output = hbam.cosine(x, y, w)

        self.assertAlmostEqual(output, 0.9128709291752769)

    def test_input_arrays_different_shape(self):
        x = np.ones(10)
        y = np.ones(11)

        with self.assertRaises(AssertionError):
            output = hbam.cosine(x, y)

    def test_cosine_no_weight_array(self):
        x = np.ones(10)
        y = np.ones(10)

        output = hbam.cosine(x, y)

        self.assertAlmostEqual(output, 1.0)


if __name__ == '__main__':
    unittest.main()
