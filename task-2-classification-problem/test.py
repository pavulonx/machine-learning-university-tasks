# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ----------------- TEN PLIK MA POZOSTAC NIEZMODYFIKOWANY ------------------
# --------------------------------------------------------------------------

import logging
import numpy as np
import pickle
import sys
from unittest import TestCase, makeSuite, TextTestRunner, TestSuite
from content import (hamming_distance, sort_train_labels_knn,
                              p_y_x_knn, classification_error, model_selection_knn, estimate_a_priori_nb,
                              estimate_p_x_y_nb, p_y_x_nb, model_selection_nb)


EPSILON_COMA = 7
PICKLE_FILE_PATH = 'test_data.pkl'

with open(PICKLE_FILE_PATH, mode='rb') as f:
    test_data = pickle.load(f)


class TestRunner(TextTestRunner):
    def __init__(self,result=None):
        super(TestRunner, self).__init__(verbosity=2)

    def run(self):
        suite = TestSuite()
        return super(TestRunner, self).run(suite)


class TestSuite(TestSuite):
    def __init__(self):
        super(TestSuite, self).__init__()
        self.addTest(makeSuite(TestHamming))
        self.addTest(makeSuite(TestSortTrainLabelsKNN))
        self.addTest(makeSuite(TestPYXKNN))
        self.addTest(makeSuite(TestClassificationError))
        self.addTest(makeSuite(TestModelSelectionKNN))
        self.addTest(makeSuite(TestEstimateAPrioriNB))
        self.addTest(makeSuite(TestEstimatePXYNB))
        self.addTest(makeSuite(TestPYXNB))
        self.addTest(makeSuite(TestModelSelectionNB))


class TestHamming(TestCase):
    def test_hamming_distance(self):
        data = test_data['hamming_distance']
        out = hamming_distance(data['X'], data['X_train'])
        self.assertTrue((out == data['Dist']).all())


class TestSortTrainLabelsKNN(TestCase):
    def test_sort_train_labels_knn(self):
        data = test_data['sort_train_labels_KNN']
        out = sort_train_labels_knn(data['Dist'], data['y'])
        self.assertTrue((out == data['y_sorted']).all())


class TestPYXKNN(TestCase):
    def test_p_y_x_knn(self):
        data = test_data['p_y_x_KNN']
        out = p_y_x_knn(data['y'], data['K'])
        max_diff = np.max(np.abs(data['p_y_x'] - out))
        self.assertAlmostEqual(max_diff, 0, 8)


class TestClassificationError(TestCase):
    def test_classification_error(self):
        data = test_data['error_fun']
        out = classification_error(data['p_y_x'], data['y_true'])
        self.assertAlmostEqual(out, data['error_val'], 8)


class TestModelSelectionKNN(TestCase):
    def test_model_selection_knn_best_error(self):
        data = test_data['model_selection_KNN']

        out = model_selection_knn(data['Xval'], data['Xtrain'], data['yval'], data['ytrain'], data['K_values'])
        self.assertAlmostEquals(out[0], data['error_best'], 8)

    def test_model_selection_knn_best_k(self):
        data = test_data['model_selection_KNN']

        out = model_selection_knn(data['Xval'], data['Xtrain'], data['yval'], data['ytrain'], data['K_values'])
        self.assertEquals(out[1], data['best_K'])

    def test_model_selection_knn_errors(self):
        data = test_data['model_selection_KNN']

        out = model_selection_knn(data['Xval'], data['Xtrain'], data['yval'], data['ytrain'], data['K_values'])

        max_diff = np.max(np.abs(data['errors'] - out[2]))
        self.assertAlmostEqual(max_diff, 0, 8)


class TestEstimateAPrioriNB(TestCase):
    def test_estimate_a_priori_nb(self):
        data = test_data['estimate_a_priori_NB']
        out = estimate_a_priori_nb(data['ytrain'])
        max_diff = np.max(np.abs(data['p_y'] - out))
        self.assertAlmostEqual(max_diff, 0, 8)


class TestEstimatePXYNB(TestCase):
    def test_estimate_p_x_y_nb(self):
        data = test_data['estimate_p_x_y_NB']
        out = estimate_p_x_y_nb(data['Xtrain'], data['ytrain'], data['a'], data['b'])
        max_diff = np.max(np.abs(data['p_x_y'] - out))
        self.assertAlmostEqual(max_diff, 0, 8)


class TestPYXNB(TestCase):
    def test_p_y_x_nb(self):
        data = test_data['p_y_x_NB']
        out = p_y_x_nb(data['p_y'], data['p_x_1_y'], data['X'])
        max_diff = np.max(np.abs(data['p_y_x'] - out))
        self.assertAlmostEqual(max_diff, 0, 8)


class TestModelSelectionNB(TestCase):
    def test_model_selection_nb_best_error(self):
        data = test_data['model_selection_NB']
        error_best, best_a, best_b, errors = model_selection_nb(data['Xtrain'], data['Xval'], data['ytrain'],
                                                                data['yval'], data['a_values'], data['b_values'])
        expected_error_best, expected_best_a, expected_best_b, expected_errors = \
            data['error_best'], data['best_a'], data['best_b'], data['errors']
        self.assertAlmostEqual(error_best, expected_error_best)

    def test_model_selection_nb_best_a(self):
        data = test_data['model_selection_NB']
        error_best, best_a, best_b, errors = model_selection_nb(data['Xtrain'], data['Xval'], data['ytrain'],
                                                                data['yval'], data['a_values'], data['b_values'])
        expected_error_best, expected_best_a, expected_best_b, expected_errors = \
            data['error_best'], data['best_a'], data['best_b'], data['errors']
        self.assertEquals(best_a, expected_best_a)

    def test_model_selection_nb_best_b(self):
        data = test_data['model_selection_NB']
        error_best, best_a, best_b, errors = model_selection_nb(data['Xtrain'], data['Xval'], data['ytrain'],
                                                                data['yval'], data['a_values'], data['b_values'])
        expected_error_best, expected_best_a, expected_best_b, expected_errors = \
            data['error_best'], data['best_a'], data['best_b'], data['errors']
        self.assertEquals(best_b, expected_best_b)

    def test_model_selection_nb_errors(self):
        data = test_data['model_selection_NB']
        error_best, best_a, best_b, errors = model_selection_nb(data['Xtrain'], data['Xval'], data['ytrain'],
                                                                data['yval'], data['a_values'], data['b_values'])
        expected_error_best, expected_best_a, expected_best_b, expected_errors = \
            data['error_best'], data['best_a'], data['best_b'], data['errors']

        max_diff = np.max(np.abs(expected_errors - errors))
        self.assertAlmostEqual(max_diff, 0, 8)