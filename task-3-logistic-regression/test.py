# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ----------------- TEN PLIK MA POZOSTAC NIEZMODYFIKOWANY ------------------
# --------------------------------------------------------------------------

import unittest
from unittest import makeSuite
import pickle
import numpy as np
from content import sigmoid
from content import logistic_cost_function
from content import regularized_logistic_cost_function
from content import gradient_descent
from content import stochastic_gradient_descent
from content import prediction
from content import f_measure
from content import model_selection
import sys
import content
sys.modules['zadanie3'] = content
sys.modules['zadanie3.content'] = content

TEST_DATA = pickle.load(open('test_data.pkl', mode='rb'))


class TestRunner(unittest.TextTestRunner):
    def __init__(self, result=None):
        super(TestRunner, self).__init__(verbosity=2)

    def run(self):
        suite = TestSuite()
        return super(TestRunner, self).run(suite)


class TestSuite(unittest.TestSuite):
    def __init__(self):
        super(TestSuite, self).__init__()
        self.addTest(makeSuite(TestSigmoid))
        self.addTest(makeSuite(TestLogisticCostFunction))
        self.addTest(makeSuite(TestGradientDescent))
        self.addTest(makeSuite(TestStochasticGradientDescent))
        self.addTest(makeSuite(TestRegularizedLogisticCostFunction))
        self.addTest(makeSuite(TestPrediction))
        self.addTest(makeSuite(TestFMeasure))
        self.addTest(makeSuite(TestModelSelection))


class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        arg = TEST_DATA['sigm']['arg']
        val = TEST_DATA['sigm']['val']
        val_computed = sigmoid(arg)
        max_diff = np.max(np.abs(val - val_computed))
        self.assertAlmostEqual(max_diff, 0, 8)


class TestLogisticCostFunction(unittest.TestCase):
    def test_logstic_cost_function_val(self):
        x_train = TEST_DATA['cost']['x_train']
        y_train = TEST_DATA['cost']['y_train']
        w = TEST_DATA['cost']['w']
        val = TEST_DATA['cost']['L']
        val_computed, _ = logistic_cost_function(w, x_train, y_train)
        self.assertAlmostEqual(val, val_computed, 6)

    def test_logstic_cost_function_grad(self):
        x_train = TEST_DATA['cost']['x_train']
        y_train = TEST_DATA['cost']['y_train']
        w = TEST_DATA['cost']['w']
        grad = TEST_DATA['cost']['grad']
        _, grad_computed = logistic_cost_function(w, x_train, y_train)
        max_diff = np.max(np.abs(grad - grad_computed))
        self.assertAlmostEqual(max_diff, 0, 6)


class TestGradientDescent(unittest.TestCase):
    def test_gradient_descent_w(self):
        w0 = np.copy(TEST_DATA['opt']['w0'])
        eta = TEST_DATA['opt']['step']
        epochs = TEST_DATA['opt']['epochs']
        w = TEST_DATA['opt']['w']
        obj_fun = TEST_DATA['opt']['obj_fun']
        w_computed, _ = gradient_descent(obj_fun, w0, epochs, eta)
        max_diff = np.max(np.abs(w - w_computed))
        self.assertAlmostEqual(max_diff, 0, 6)

    def test_gradient_descent_func_values(self):
        w0 = np.copy(TEST_DATA['opt']['w0'])
        eta = TEST_DATA['opt']['step']
        epochs = TEST_DATA['opt']['epochs']
        func_values = TEST_DATA['opt']['func_values']
        obj_fun = TEST_DATA['opt']['obj_fun']
        _, func_values_computed = gradient_descent(obj_fun, w0, epochs, eta)
        max_diff = np.max(np.abs(func_values - func_values_computed))
        self.assertAlmostEqual(max_diff, 0, 6)


class TestStochasticGradientDescent(unittest.TestCase):
    def test_stochastic_gradient_descent_w(self):
        x_train = TEST_DATA['sopt']['x_train']
        y_train = TEST_DATA['sopt']['y_train']
        w0 = np.copy(TEST_DATA['sopt']['w0'])
        eta = TEST_DATA['sopt']['step']
        epochs = TEST_DATA['sopt']['epochs']
        w = TEST_DATA['sopt']['w']
        obj_fun = TEST_DATA['sopt']['obj_fun']
        mini_batch = TEST_DATA['sopt']['mini_batch']
        w_computed, _ = stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch)
        max_diff = np.max(np.abs(w - w_computed))
        self.assertAlmostEqual(max_diff, 0, 6)

    def test_stochastic_gradient_descent_func_values(self):
        x_train = TEST_DATA['sopt']['x_train']
        y_train = TEST_DATA['sopt']['y_train']
        w0 = np.copy(TEST_DATA['sopt']['w0'])
        eta = TEST_DATA['sopt']['step']
        epochs = TEST_DATA['sopt']['epochs']
        func_values = TEST_DATA['sopt']['func_values']
        obj_fun = TEST_DATA['sopt']['obj_fun']
        mini_batch = TEST_DATA['sopt']['mini_batch']
        _, func_values_computed = stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch)
        max_diff = np.max(np.abs(func_values - func_values_computed))
        self.assertAlmostEqual(max_diff, 0, 6)


class TestRegularizedLogisticCostFunction(unittest.TestCase):
    def test_regularized_logstic_cost_function_val(self):
        x_train = TEST_DATA['rcost']['x_train']
        y_train = TEST_DATA['rcost']['y_train']
        w = TEST_DATA['rcost']['w']
        reg_lambda = TEST_DATA['rcost']['lambda']
        val = TEST_DATA['rcost']['L']
        val_computed, _ = regularized_logistic_cost_function(w, x_train, y_train, reg_lambda)
        self.assertAlmostEqual(val, val_computed, 6)

    def test_regularized_logstic_cost_function_grad(self):
        x_train = TEST_DATA['rcost']['x_train']
        y_train = TEST_DATA['rcost']['y_train']
        w = TEST_DATA['rcost']['w']
        reg_lambda = TEST_DATA['rcost']['lambda']
        grad = TEST_DATA['rcost']['grad']
        _, grad_computed = regularized_logistic_cost_function(w, x_train, y_train, reg_lambda)
        max_diff = np.max(np.abs(grad - grad_computed))
        self.assertAlmostEqual(max_diff, 0, 6)


class TestPrediction(unittest.TestCase):
    def test_prediction(self):
        x = TEST_DATA['pred']['x']
        w = TEST_DATA['pred']['w']
        theta = TEST_DATA['pred']['theta']
        y = TEST_DATA['pred']['y']
        y_computed = prediction(x, w, theta)
        max_diff = np.max(np.abs(y - y_computed))
        self.assertAlmostEqual(max_diff, 0, 6)


class TestFMeasure(unittest.TestCase):
    def test_f_measure(self):
        y = TEST_DATA['fm']['y']
        y_pred = TEST_DATA['fm']['y_pred']
        f = TEST_DATA['fm']['f']
        f_computed = f_measure(y,y_pred)
        self.assertAlmostEqual(f, f_computed, 6)


class TestModelSelection(unittest.TestCase):
    def test_model_selection_lambda(self):
        x_train = TEST_DATA['ms']['x_train']
        y_train = TEST_DATA['ms']['y_train']
        x_val = TEST_DATA['ms']['x_val']
        y_val = TEST_DATA['ms']['y_val']
        w0 = TEST_DATA['ms']['w0']
        eta = TEST_DATA['ms']['step']
        epochs = TEST_DATA['ms']['epochs']
        mini_batch = TEST_DATA['ms']['mini_batch']
        thetas = TEST_DATA['ms']['thetas']
        lambdas = TEST_DATA['ms']['lambdas']
        reg_lambda = TEST_DATA['ms']['lambda']
        reg_lambda_computed,_,_,_ = model_selection(x_train,y_train,x_val,y_val,w0,epochs,eta,mini_batch,lambdas,thetas)
        self.assertAlmostEqual(reg_lambda, reg_lambda_computed, 6)

    def test_model_selection_theta(self):
        x_train = TEST_DATA['ms']['x_train']
        y_train = TEST_DATA['ms']['y_train']
        x_val = TEST_DATA['ms']['x_val']
        y_val = TEST_DATA['ms']['y_val']
        w0 = TEST_DATA['ms']['w0']
        eta = TEST_DATA['ms']['step']
        epochs = TEST_DATA['ms']['epochs']
        mini_batch = TEST_DATA['ms']['mini_batch']
        thetas = TEST_DATA['ms']['thetas']
        lambdas = TEST_DATA['ms']['lambdas']
        theta = TEST_DATA['ms']['theta']
        _,theta_computed,_,_ = model_selection(x_train,y_train,x_val,y_val,w0,epochs,eta,mini_batch,lambdas,thetas)
        self.assertAlmostEqual(theta, theta_computed, 6)

    def test_model_selection_w(self):
        x_train = TEST_DATA['ms']['x_train']
        y_train = TEST_DATA['ms']['y_train']
        x_val = TEST_DATA['ms']['x_val']
        y_val = TEST_DATA['ms']['y_val']
        w0 = TEST_DATA['ms']['w0']
        eta = TEST_DATA['ms']['step']
        epochs = TEST_DATA['ms']['epochs']
        mini_batch = TEST_DATA['ms']['mini_batch']
        thetas = TEST_DATA['ms']['thetas']
        lambdas = TEST_DATA['ms']['lambdas']
        w = TEST_DATA['ms']['w']
        _,_,w_computed,_ = model_selection(x_train,y_train,x_val,y_val,w0,epochs,eta,mini_batch,lambdas,thetas)
        max_diff = np.max(np.abs(w - w_computed))
        self.assertAlmostEqual(max_diff, 0, 6)

    def test_model_selection_F(self):
        x_train = TEST_DATA['ms']['x_train']
        y_train = TEST_DATA['ms']['y_train']
        x_val = TEST_DATA['ms']['x_val']
        y_val = TEST_DATA['ms']['y_val']
        w0 = TEST_DATA['ms']['w0']
        eta = TEST_DATA['ms']['step']
        epochs = TEST_DATA['ms']['epochs']
        mini_batch = TEST_DATA['ms']['mini_batch']
        thetas = TEST_DATA['ms']['thetas']
        lambdas = TEST_DATA['ms']['lambdas']
        F = TEST_DATA['ms']['F']
        _,_,_,F_computed = model_selection(x_train,y_train,x_val,y_val,w0,epochs,eta,mini_batch,lambdas,thetas)
        max_diff = np.max(np.abs(F - F_computed))
        self.assertAlmostEqual(max_diff, 0, 6)