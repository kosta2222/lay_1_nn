from nn_constants import elems_of_img, max_trainSet_rows
import numpy as np
import os
from PIL import Image
def make_train_matr(p_:str)->np.ndarray:
    matr=np.zeros(shape=(max_trainSet_rows, elems_of_img))
    data=None
    img=None
    for i in os.listdir(p_):
        ful_p=os.path.join(p_,i)
        img=Image.open(ful_p)
        print("img", ful_p)
        data=list(img.getdata())
        for row in range(max_trainSet_rows):
            for elem in range(elems_of_img):
                matr[row][elem] = data[elem]
    return matr
