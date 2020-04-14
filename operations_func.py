from nn_constants import RELU_DERIV, RELU, TRESHOLD_FUNC, TRESHOLD_FUNC_DERIV, LEAKY_RELU, LEAKY_RELU_DERIV,\
SIGMOID, SIGMOID_DERIV, DEBUG, DEBUG_STR, INIT_W_HE, INIT_W_GLOROT, INIT_W_HABR, INIT_W_MY
import numpy as np
import math
np.random.seed(42)
ready = False


# операции для функций активаций и их производных - для numpy массивов
def operations(op, a, b, c, d, str= ""):
    global ready
    a=a.T
    a=a[0]
    """
    В основном для функций активаций
    :param op: 'байт-комманда'
    :param a: <>
    :param b: <>
    :param c: <>
    :param d: <>
    :param str: <>-ее параметры
    :return:
    """
    l=[]
    if op==RELU:
        for i in a:
            if (i < 0):
                l.append(0)
            else:
                l.append(i)
        return np.array([l]).T
    elif op==RELU_DERIV:
        for i in a:
            if (i < 0):
                l.append(0)
            else:
                l.append(1)
        return np.array([l])
    elif op==TRESHOLD_FUNC:
        for i in a:
            if (i < 0):
                l.append(0)
            else:
                l.append(1)
        return np.array([l]).T
    elif op==TRESHOLD_FUNC_DERIV:
        pass # Нет производной
    elif op==LEAKY_RELU:
        for i in a:
            if (i < 0):
                l.append(b * a)
            else:
                l.append(i)
        return np.array([l]).T
    elif op==LEAKY_RELU_DERIV:
        for i in a:
            if (i < 0):
                l.append(b)
            else:
                l.append(1)
        return np.array([l]).T
    elif op==SIGMOID:
        for i in a:
            s = 1.0 / (1 + np.exp(b * (- i)))
            l.append(s)
        return np.array([l]).T
    elif op==SIGMOID_DERIV:
        for i in a:
            s = (b * 1.0 / (1 + np.exp(b * (- i))) * (1 - 1.0 / (1 + np.exp(b * (- i)))))
            l.append(s)
        return np.array([l]).T
    elif op==DEBUG:
        print("%s : %f"% str, a)
    elif op==DEBUG_STR:
        print("%s"% str)
    elif op == INIT_W_HE:
        return np.random.randn() * math.sqrt(2 / b)
    elif op == INIT_W_MY:
        if ready:
            ready = False
            return -0.01
        ready = True
        return 0.01

