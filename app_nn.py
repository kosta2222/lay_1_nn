from operations_func import operations
from read_x_y import make_train_matr
from nn_constants import  INIT_W_HE, RELU,RELU_DERIV, \
    INIT_W_HABR, INIT_W_MY, SIGMOID, SIGMOID_DERIV, max_trainSet_rows, elems_of_img, max_rows_orOut
import numpy as np
def init_wei(w,h)->np.ndarray:
    matr = np.zeros((h, w))
    for row in range(h):
        for elem in range(w):
            matr[row][elem] = operations(INIT_W_HE, np.array([[0]]), 10000, 0, 0, "");
    return matr
def main():
    wei:np.ndarray=None
    n1:np.ndarray=None
    data:np.ndarray=None
    answer:np.ndarray=None
    answer=np.array([[1, 1, 1, 1]])
    wei = init_wei(elems_of_img, max_rows_orOut)
    print("wei shape",wei.shape)
    # data=[[0,0],[1,0],[0,1],[1,1]]
    # answer=[0, 1, 1, 1]  #  OR
    # answer=[[0, 0, 0, 1]]  # AND
    # n1=[0]*2
    # n2=0
    n2:np.ndarray=None
    w2=[0]*3
    n2_dot:np.ndarray=None
    count=0
    A=0.07
    E=0
    E1=0
    E2=0
    E3=0
    choose=0
    choose_cv=0
    eps=10
    mse=0
    exit_flag=False
    scores=[]
    theme=""
    theme="AND"
    # theme="OR"
    alpha=0.99
    beta=1.01
    gama=1.01
    delta_E_spec=0
    Z=0
    Z_t_minus_1=0
    A_t_minus_1=0
    acc=0
    sigmoid_koef=0.42
    accuracy_shureness = 25
    with_adap_lr = False
    data = make_train_matr("b:/out")
    while (1) :
        print("epocha %d\n" % count);
        # Обучение (вмести с утверждением крос-валидации)
        while (choose <= 3):
            print("chose %d \n" % choose)

            print("data shape",data.shape)  # X как (4, 10_000)(?!)
            n1 = np.array([data[choose]])  # входные данные как (1, 10_000)
            print("n1 shape",n1.shape)
            """
            Умножаю значения нейронов 1 слоя с соответствующими весами и
            пропускаю через функцию активации которая является сигмоидом 
            """
            n2_dot=np.dot(wei, n1.T)
            print("n2_dot shape",n2_dot.shape)

            n2 = operations(RELU, n2_dot, 1, 0, 0, "")
            # Получаю ошибку выходного нейрона
            Z = n2 - answer.T[choose]
            # print("Z",Z)
            E = (n2 - answer.T[choose]) * operations(RELU_DERIV, n2_dot, 1, 0, 0, "");
            if count == 0:
                Z_t_minus_1 = Z[0][0]
                A_t_minus_1 = A
            mse = pow(answer.T[choose] - n2, 2);
            print("mse in train: %f \n" % mse);
            # if (mse < 0.0001):
            #     print("op")
            #     exit_flag=True
            #     break
            if with_adap_lr:
                delta_E_spec = Z[0][0] - gama * Z_t_minus_1
                if delta_E_spec > 0:
                    A = alpha * A_t_minus_1
                else:
                    A = beta * A_t_minus_1
                print("A",A)
            A_t_minus_1 = A
            Z_t_minus_1 = Z[0][0]
            E1 = E * wei
            wei = wei - A * E1 * n1
            # Выход по крос-валидации
            choose_cv = 0
            while (choose_cv <= 3):
                # data = make_train_matr("b:/out")
                n1 = np.array([data[choose]])  # входные данные как (1, 10_000)
                n2_dot = np.dot(wei, n1.T)
                n2 = operations(RELU, n2_dot, 1, 0, 0, "")
                if (n2[0][0] > 0.5):
                    n2[0][0] = 1
                    print("output vector[ %f ] " % 1, end=' ')
                else:
                    n2[0][0] = 0
                    print("output vector[ %f ] " % 0, end=' ')
                print("expected [ %f ]\n" % answer.T[choose_cv])
                if n2[0][0] == answer[0][choose_cv]:
                    scores.append(1)
                else:
                    scores.append(0)
                choose_cv+=1
            # choose_cv = 0
            acc = sum(scores) / 4 * 100
            print("Accuracy statement",acc)
            scores.clear()
            if acc == accuracy_shureness:
                exit_flag = True
                break
            choose+=1
        choose = 0
        count+=1
        if exit_flag:
            break
    # scores.clear()
    """
    # Сеть
    # обучилась - проведем
    # консольную
    # кросс - валидацию

    print("***Cons Cv - %s***\n" % theme);
    choose = 0;
    #

    #
    print("input vector [ %f %f ] " % (n1[0], n1[1]));
    if (n2 > 0.5):
        n2 = 1
        print("output vector[ %f ] " % 1, end=' ')
    else:
        n2 = 0
        print("output vector[ %f ] " % 0, end=' ');
    print("expected [ %f ]\n" % answer[choose]);
    if n2 == answer[choose]:
        scores.append(1)
    else:
        scores.append(0)

    choose += 1;
    acc = sum(scores) / 4 * 100
    print("Accuracy", acc)
    """




main()