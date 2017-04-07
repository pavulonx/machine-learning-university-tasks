# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np
import scipy.spatial.distance as dist


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """

    X = X.toarray()
    X_train = X_train.toarray()

    to_return = dist.cdist(X, X_train, "hamming")
    to_return = (to_return * X.shape[1]).astype(int)
    return to_return


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    indices_to_sort = np.argsort(Dist, kind="mergesort")
    ret = np.fromfunction(lambda i, j: y[indices_to_sort[i, j]], shape=Dist.shape, dtype=int)
    return ret


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    number_of_classes = np.unique(y[0, :]).shape[0]
    # number_of_classes = 4

    y = y[:, :k]
    ret = np.zeros(shape=(y.shape[0], number_of_classes), dtype=float)

    for j in range(ret.shape[0]):
        for i in range(ret.shape[1]):
            ret[j, i] = np.count_nonzero((y[j, :]) == (i + 1))

    ret = ret / k
    return ret


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    N1 = p_y_x.shape[0]
    result = 0
    rev_p_y_x = np.fliplr(p_y_x)
    to_comp = 4 - np.argmax(a=rev_p_y_x, axis=1)
    result = to_comp - y_true
    return (np.count_nonzero(result)) / N1


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """

    distance = hamming_distance(Xval, Xtrain)
    y_sorted = sort_train_labels_knn(distance, ytrain)
    errors = []
    for k in k_values:
        error_k = classification_error(p_y_x_knn(y_sorted, k), yval)
        errors.append(error_k)
    best_error = min(errors)  # min error
    best_k = k_values[errors.index(best_error)]
    return best_error, best_k, errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    N = ytrain.shape[0]
    res = np.zeros(shape=(4))
    for i in range(1, 5):
        res[i - 1] = np.count_nonzero(ytrain == i) / N
    return res
    # p-stwo y-tej klasy


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """

    Xtrain = Xtrain.toarray()
    up_factor = a - 1.0
    down_factor = a + b - 2.0

    def f(k, d):
        I_yn_k = (ytrain == k + 1).astype(bool)
        I_xnd_1 = (Xtrain[:, d] == 1).astype(bool)
        up = up_factor + np.count_nonzero(I_yn_k & I_xnd_1)
        down = down_factor + np.count_nonzero(I_yn_k)
        return up / down

    g = np.vectorize(f)
    return np.fromfunction(g, shape=(4, Xtrain.shape[1]), dtype=int)
    # theta{d,k}


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    # wwolne
    #
    # N = np.shape(X)[0]
    # D = np.shape(X)[1]
    # M = np.shape(p_y)[0]
    #
    # result = np.ones(shape=(N, M))
    #
    # for n in range(N):
    #     denominator = 0.0
    #     for m in range(M):
    #         for d in range(D):
    #             if X[n, d] == 1:
    #                 result[n, m] *= p_x_1_y[m, d]
    #             else:
    #                 result[n, m] *= 1 - p_x_1_y[m, d]
    #         result[n, m] *= p_y[m]
    #         denominator += result[n, m]
    #     result[n] /= denominator
    # return result

    # najlepsze

    N = np.shape(X)[0]
    M = np.shape(p_y)[0]
    X = X.toarray()

    def f(n, m):
        return np.prod(np.negative(X[n, :]) - p_x_1_y[m, :])

    g = np.vectorize(f)
    result = np.fromfunction(g, shape=(N, M), dtype=int) * p_y
    result /= result @ np.ones(shape=(4, 1))

    return result


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """

    A = len(a_values)
    B = len(b_values)

    p_y = estimate_a_priori_nb(ytrain)

    def f(a, b):
        p_x_y_nb = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a], b_values[b])
        p_y_x = p_y_x_nb(p_y, p_x_y_nb, Xval)
        err = classification_error(p_y_x, yval)
        return err

    g = np.vectorize(f)
    errors = np.fromfunction(g, shape=(A, B), dtype=int)

    minimum = np.argmin(errors)
    minA = minimum // A
    minB = minimum % A
    return errors[minA, minB], a_values[minA], b_values[minB], errors
