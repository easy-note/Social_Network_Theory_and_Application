
import numpy as np
import numpy.linalg as lin
from numpy.linalg import norm

# np.set_printoptions(linewidth=np.inf) 
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})


def cal_cosine(target_1, target_2):
    target_1, target_2 = np_preprocessing(target_1, target_2)
    return np.dot(target_1,target_2)/(norm(target_1)*norm(target_2))

def np_preprocessing(target_1, target_2):
    t1 = []
    t2 = []
    for (a, b) in zip(target_1, target_2):
        if ((a) and (b)):
            t1.append(a)
            t2.append(b)

    return np.array(t1), np.array(t2)


r_mean = 3.83

'''
 nan = -1000
 test = -100
'''
r = np.array([
    [5,4,4,-1000,-100],
    [-1000,3,5,-100,4],
    [5,2,-1000,-100,3],
    [-1000,-100,3,1,2],
    [4,-1000,-100,4,5],
    [-100,3,-1000,3,5],
    [3,-100,3,2,-1000],
    [5,-100,4,-1000,5],
    [-100,2,5,4,-1000],
    [-100,-1000,5,3,4]]
    )

# a matrix 생성
a = np.zeros((30, 15))

cnt = 0
for col in range(5):
    for row, t in enumerate(r):
        if t[col] > 0:
            # print(col+1, '열', row+1, '행')

            a[cnt][row] = 1
            a[cnt][10+col] = 1
            cnt += 1


# c vector 생성
c = np.array([5,5,4,3,5,4,3,2,3,2,4,5,3,3,4,5,5,1,4,3,2,4,3,4,3,2,5,5,5,4])
c = c-r_mean

b = np.dot(lin.pinv(a), c)

# r_hat matrix 생성
r_hat = np.full((10, 5), r_mean)

for idx1, row in enumerate(r_hat):
    for idx2, col in enumerate(row):

        r_hat[idx1][idx2] += b[idx1]
        r_hat[idx1][idx2] += b[10+idx2]

# print('r_hat')
# print(r_hat)

# r_tilda matrix 생성
r_tilda = r - r_hat
r_tilda[r_tilda < -50] = False
# print('r_tilda')
# print(r_tilda)

# d matrix 계산
d = np.full((5, 5), -1.000)

for idx1, row in enumerate(d):
    for idx2, col in enumerate(row):
        if idx1 != idx2:
            target_1 = r_tilda[:,idx1]
            target_2 = r_tilda[:,idx2]

            cosine_score = cal_cosine(target_1, target_2)
        
            d[idx1][idx2] = cosine_score

# d matrix - train data 만 남기기
d[d == -1.000] = None
print('**** Answer ****')
print(d)



