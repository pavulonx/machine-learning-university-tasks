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
    return 1/(1+np.exp(-x))


def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej, a grad jej gradient po w
    '''
    pass


def gradient_descent(obj_fun, w0, epochs, eta):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w).
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok / iteracji algorytmu
    :param eta: krok uczenia
    :return: funkcja wykonuje optymalizacje metoda gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_valus jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu
    '''
    pass


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
    pass


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej z regularyzacja l2,
    a grad jej gradient po w
    '''
    pass


def prediction(x, w, theta):
    '''
    :param x: macierz obserwacji NxM
    :param w: wektor parametrow modelu Mx1
    :param theta: prog klasyfikacji z przedzialu [0,1]
    :return: funkcja wylicza wektor y o wymiarach Nx1. Wektor zawiera wartosci etykiet ze zbioru {0,1} dla obserwacji z x
     bazujac na modelu z parametrami w oraz progu klasyfikacji theta
    '''
    pass


def f_measure(y_true, y_pred):
    '''
    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet przewidzianych przed model Nx1
    :return: funkcja wylicza wartosc miary F
    '''
    pass


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
    pass
