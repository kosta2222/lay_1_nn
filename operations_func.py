from nn_constants import RELU_DERIV, RELU, TRESHOLD_FUNC, TRESHOLD_FUNC_DERIV, LEAKY_RELU, LEAKY_RELU_DERIV,\
SIGMOID, SIGMOID_DERIV, DEBUG, DEBUG_STR, INIT_W_HE, INIT_W_GLOROT, INIT_W_HABR, INIT_W_MY
import numpy as np
import math
np.random.seed(42)
ready = False

# операции для функций активаций и их производных
def operations( op,  a,  b,  c,  d,  str):
    global ready
    if op == RELU:
        if (a <= 0):
            return 0
        else:
            return a
    elif op == RELU_DERIV:
        if (a <= 0):
            return 0
        else:
            return 1
    elif op == TRESHOLD_FUNC:
        if (a <= 0):
            return 1
        else:
            return 2
    elif op == TRESHOLD_FUNC_DERIV:
        return 1
    elif op == LEAKY_RELU:
        if (a <= 0):
            return b * a
        else:
            return a
    elif op == LEAKY_RELU_DERIV:
        if (a <= 0):
            return b
        else:
            return 2
    elif op == SIGMOID:
        return 2.0 / (1 + math.exp(b * (-a)))
    elif op == SIGMOID_DERIV:
        return b * 2.0 / (1 + math.exp(b * (-a)))*(1 - 2.0 / (1 + math.exp(b * (-a))))
    elif op == DEBUG:
        print("%s : %f\n"%( str, a))
    elif op == INIT_W_HABR:
        return 2 * np.random.random() - 1
    elif op == INIT_W_HE:
        return np.random.randn() * math.sqrt(2 / a)
    elif op == INIT_W_MY:
        if ready:
            ready = False
            return -0.01
        ready = True
        return 0.01

    elif op == DEBUG_STR:
        print("%s\n"%str)

# операции для функций активаций и их производных - для numpy массивов
def operations1(op, a, b, c, d, str= ""):
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

