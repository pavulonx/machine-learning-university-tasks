# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np


def sigmoid(x):
    '''
    :param x: wektor wejsciowych wartosci Nx1
    :return: wektor wyjściowych wartości funkcji sigmoidalnej dla wejścia x, Nx1
    '''
    return 1 / (1 + np.exp(-x))


def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej, a grad jej gradient po w
    '''
    sigma = sigmoid(x_train @ w)
    N = y_train.shape[0]
    p_D_w = np.prod((sigma ** y_train) * ((1 - sigma) ** (1 - y_train)))
    L_w = -np.log(p_D_w) / N
    grad = - (x_train.transpose() @ (y_train - sigma)) / N
    return L_w, grad


def gradient_descent(obj_fun, w0, epochs, eta):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w).
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok / iteracji algorytmu
    :param eta: krok uczenia
    :return: funkcja wykonuje optymalizacje metoda gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_valus jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu
    '''
    w = w0
    wA = []
    _, grad = obj_fun(w0)
    for i in range(epochs):
        w = w - eta * grad
        val, grad = obj_fun(w)
        wA.append(val)
    return w, np.array(wA).reshape(epochs, 1)


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w,x,y), gdzie x,y oznaczaja podane
    podzbiory zbioru treningowego (mini-batche)
    :param x_train: dane treningowe wejsciowe NxM
    :param y_train: dane treningowe wyjsciowe Nx1
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini-batcha
    :return: funkcja wykonuje optymalizacje metoda stochastycznego gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_values jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu. Wartosci
    funkcji do func_values sa wyliczane dla calego zbioru treningowego!
    '''
    N = y_train.shape[0]
    M = N // mini_batch
    x_mini_batch = np.vsplit(x_train, M)
    y_mini_batch = np.vsplit(y_train, M)

    w = w0
    wA = []
    for i in range(epochs):
        for x, y in zip(x_mini_batch, y_mini_batch):
            _, grad = obj_fun(w, x, y)
            w = w - eta * grad
        wA.append(obj_fun(w, x_train, y_train)[0])
    return w, np.array(wA).reshape(epochs, 1)


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej z regularyzacja l2,
    a grad jej gradient po w
    '''
    w_0 = np.copy(w)
    w_0[0] = 0
    N = y_train.shape[0]
    sigma = sigmoid(x_train @ w)
    p_D_w = np.prod((sigma ** y_train) * ((1 - sigma) ** (1 - y_train)))
    L_w_l = -np.log(p_D_w) / N + regularization_lambda / 2 * np.linalg.norm(w_0) ** 2
    grad = - (x_train.transpose() @ (y_train - sigma)) / N + regularization_lambda * w_0
    return L_w_l, grad


def prediction(x, w, theta):
    '''
    :param x: macierz obserwacji NxM
    :param w: wektor parametrow modelu Mx1
    :param theta: prog klasyfikacji z przedzialu [0,1]
    :return: funkcja wylicza wektor y o wymiarach Nx1. Wektor zawiera wartosci etykiet ze zbioru {0,1} dla obserwacji z x
     bazujac na modelu z parametrami w oraz progu klasyfikacji theta
    '''
    return np.apply_along_axis(arr=sigmoid(x @ w), func1d=lambda arg: arg >= theta, axis=1)


def f_measure(y_true, y_pred):
    '''
    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet przewidzianych przed model Nx1
    :return: funkcja wylicza wartosc miary F
    '''
    N = y_true.shape[0]
    TP = np.count_nonzero(y_true & y_pred)
    F = np.count_nonzero(np.bitwise_xor(y_true, y_pred))
    return 2 * TP / (2 * TP + F)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    '''
    :param x_train: ciag treningowy wejsciowy NxM
    :param y_train: ciag treningowy wyjsciowy Nx1
    :param x_val: ciag walidacyjny wejsciowy Nval x M
    :param y_val: ciag walidacyjny wyjsciowy Nval x 1
    :param w0: wektor poczatkowych wartosci parametrow
    :param epochs: liczba epok dla SGD
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini batcha
    :param lambdas: lista wartosci parametru regularyzacji lambda, ktore maja byc sprawdzone
    :param thetas: lista wartosci progow klasyfikacji theta, ktore maja byc sprawdzone
    :return: funckja wykonuje selekcje modelu. Zwraca krotke (regularization_lambda, theta, w, F), gdzie regularization_lambda
    to najlpszy parametr regularyzacji, theta to najlepszy prog klasyfikacji, a w to najlepszy wektor parametrow modelu.
    Dodatkowo funkcja zwraca macierz F, ktora zawiera wartosci miary F dla wszystkich par (lambda, theta). Do uczenia nalezy
    korzystac z algorytmu SGD oraz kryterium uczenia z regularyzacja l2.
    '''
    F = np.zeros(shape=(len(lambdas), len(thetas)))
    max_f = [0, 0, 0, -1]
    for i, λ in enumerate(lambdas):
        obj_fun = lambda w, x, y: regularized_logistic_cost_function(w, x, y, λ)
        w, func_values = stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch)
        for j, θ in enumerate(thetas):
            f = f_measure(y_val, prediction(x_val, w, θ))
            F[i, j] = f
            if f > max_f[3]: max_f = [λ, θ, w, f]
    max_f[3] = F
    return tuple(max_f)