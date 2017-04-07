# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ----------------- TEN PLIK MA POZOSTAC NIEZMODYFIKOWANY ------------------
# --------------------------------------------------------------------------

import unittest
import pickle
import numpy as np
import sys
from content import mean_squared_error
from content import design_matrix
from content import least_squares
from content import regularized_least_squares
from content import model_selection
from content import regularized_model_selection

TEST_DATA = pickle.load(open('test.pkl', mode='rb'))


class TestRunner(unittest.TextTestRunner):
    def __init__(self,result=None):
        super(TestRunner, self).__init__(verbosity=2)

    def run(self):
        suite = TestSuite()
        return super(TestRunner, self).run(suite)


class TestSuite(unittest.TestSuite):
    def __init__(self):
        super(TestSuite, self).__init__()
        self.addTest(unittest.makeSuite(TestMeanSquaredError))
        self.addTest(unittest.makeSuite(TestDesignMatrix))
        self.addTest(unittest.makeSuite(TestLeastSquares))
        self.addTest(unittest.makeSuite(TestRegularizedLeastSquares))
        self.addTest(unittest.makeSuite(TestModelSelection))
        self.addTest(unittest.makeSuite(TestRegularizedModelSelection))


class TestMeanSquaredError(unittest.TestCase):
    def test_mean_squared_error(self):
        x = TEST_DATA['mean_error']['x']
        y = TEST_DATA['mean_error']['y']
        w = TEST_DATA['mean_error']['w']
        err = TEST_DATA['mean_error']['err']
        err_computed = mean_squared_error(x, y, w)
        self.assertAlmostEqual(err, err_computed, 8)


class TestDesignMatrix(unittest.TestCase):
    def test_design_matrix(self):
        x_train = TEST_DATA['design_matrix']['x_train']
        M = TEST_DATA['design_matrix']['M']
        dm = TEST_DATA['design_matrix']['dm']
        dm_computed = design_matrix(x_train, M)
        max_diff = np.max(np.abs(dm - dm_computed))
        self.assertAlmostEqual(max_diff, 0, 8)


class TestLeastSquares(unittest.TestCase):
    def test_least_squares_w(self):
        x_train = TEST_DATA['ls']['x_train']
        y_train = TEST_DATA['ls']['y_train']
        M = TEST_DATA['ls']['M']
        w = TEST_DATA['ls']['w']
        w_computed, _ = least_squares(x_train, y_train, M)
        max_diff = np.max(np.abs(w - w_computed))
        self.assertAlmostEqual(max_diff, 0, 6)

    def test_least_squares_err(self):
        x_train = TEST_DATA['ls']['x_train']
        y_train = TEST_DATA['ls']['y_train']
        M = TEST_DATA['ls']['M']
        err = TEST_DATA['ls']['err']
        _, err_computed = least_squares(x_train, y_train, M)
        self.assertAlmostEqual(err, err_computed, 8)


class TestRegularizedLeastSquares(unittest.TestCase):
    def test_regularized_least_squares_w(self):
        x_train = TEST_DATA['rls']['x_train']
        y_train = TEST_DATA['rls']['y_train']
        M = TEST_DATA['rls']['M']
        w = TEST_DATA['rls']['w']
        regularization_lambda = TEST_DATA['rls']['lambda']
        w_computed, _ = regularized_least_squares(x_train, y_train, M, regularization_lambda)
        max_diff = np.max(np.abs(w - w_computed))
        self.assertAlmostEqual(max_diff, 0, 6)

    def test_regularized_least_squares_err(self):
        x_train = TEST_DATA['rls']['x_train']
        y_train = TEST_DATA['rls']['y_train']
        M = TEST_DATA['rls']['M']
        err = TEST_DATA['rls']['err']
        regularization_lambda = TEST_DATA['rls']['lambda']
        _, err_computed = regularized_least_squares(x_train, y_train, M, regularization_lambda)
        self.assertAlmostEqual(err, err_computed, 8)


class TestModelSelection(unittest.TestCase):
    def test_model_selection_w(self):
        x_train = TEST_DATA['ms']['x_train']
        y_train = TEST_DATA['ms']['y_train']
        x_val = TEST_DATA['ms']['x_val']
        y_val = TEST_DATA['ms']['y_val']
        M_values = TEST_DATA['ms']['M_values']
        w = TEST_DATA['ms']['w']
        w_computed, _, _ = model_selection(x_train, y_train, x_val, y_val, M_values)
        max_diff = np.max(np.abs(w - w_computed))
        self.assertAlmostEqual(max_diff, 0, 6)

    def test_model_selection_train_err(self):
        x_train = TEST_DATA['ms']['x_train']
        y_train = TEST_DATA['ms']['y_train']
        x_val = TEST_DATA['ms']['x_val']
        y_val = TEST_DATA['ms']['y_val']
        M_values = TEST_DATA['ms']['M_values']
        train_err = TEST_DATA['ms']['train_err']
        _, train_err_computed, _ = model_selection(x_train, y_train, x_val, y_val, M_values)
        self.assertAlmostEqual(train_err, train_err_computed, 6)

    def test_model_selection_val_err(self):
        x_train = TEST_DATA['ms']['x_train']
        y_train = TEST_DATA['ms']['y_train']
        x_val = TEST_DATA['ms']['x_val']
        y_val = TEST_DATA['ms']['y_val']
        M_values = TEST_DATA['ms']['M_values']
        w = TEST_DATA['ms']['w']
        val_err = TEST_DATA['ms']['val_err']
        _, _, val_err_computed = model_selection(x_train, y_train, x_val, y_val, M_values)
        self.assertAlmostEqual(val_err, val_err_computed, 6)


class TestRegularizedModelSelection(unittest.TestCase):
    def test_regularized_model_selection_w(self):
        x_train = TEST_DATA['rms']['x_train']
        y_train = TEST_DATA['rms']['y_train']
        x_val = TEST_DATA['rms']['x_val']
        y_val = TEST_DATA['rms']['y_val']
        M = TEST_DATA['rms']['M']
        lambda_values = TEST_DATA['rms']['lambda_values']
        w = TEST_DATA['rms']['w']
        w_computed, _, _, _ = regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values)
        max_diff = np.max(np.abs(w - w_computed))
        self.assertAlmostEqual(max_diff, 0, 6)

    def test_regularized_model_selection_train_err(self):
        x_train = TEST_DATA['rms']['x_train']
        y_train = TEST_DATA['rms']['y_train']
        x_val = TEST_DATA['rms']['x_val']
        y_val = TEST_DATA['rms']['y_val']
        M = TEST_DATA['rms']['M']
        lambda_values = TEST_DATA['rms']['lambda_values']
        train_err = TEST_DATA['rms']['train_err']
        _, train_err_computed, _, _ = regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values)
        self.assertAlmostEqual(train_err, train_err_computed, 6)

    def test_regularized_model_selection_val_err(self):
        x_train = TEST_DATA['rms']['x_train']
        y_train = TEST_DATA['rms']['y_train']
        x_val = TEST_DATA['rms']['x_val']
        y_val = TEST_DATA['rms']['y_val']
        M = TEST_DATA['rms']['M']
        lambda_values = TEST_DATA['rms']['lambda_values']
        val_err = TEST_DATA['rms']['val_err']
        _, _, val_err_computed, _ = regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values)
        self.assertAlmostEqual(val_err, val_err_computed, 6)

    def test_regularized_model_selection_lambda(self):
        x_train = TEST_DATA['rms']['x_train']
        y_train = TEST_DATA['rms']['y_train']
        x_val = TEST_DATA['rms']['x_val']
        y_val = TEST_DATA['rms']['y_val']
        M = TEST_DATA['rms']['M']
        lambda_values = TEST_DATA['rms']['lambda_values']
        lambda_org = TEST_DATA['rms']['lambda']
        _, _, _, lambda_computed = regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values)
        self.assertAlmostEqual(lambda_org, lambda_computed, 6)
