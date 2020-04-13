import numpy as np
def make_train_matr(p_:str):
    matr=np.zeros(shape=(4, 10000))
    data=None
    img=None
    for i in os.listdir(p_):
        ful_p=os.path.join(p_,i)
        img=Image.open(ful_p)
        print("img", ful_p)
        data=list(img.getdata())
        for row in range(4):
            for elem in range(10000):
                matr[row][elem] = data[elem]
    return matr
