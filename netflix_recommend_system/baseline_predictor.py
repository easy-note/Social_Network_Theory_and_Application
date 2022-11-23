import numpy as np
np.set_printoptions(linewidth=np.inf) 

import numpy.linalg as lin

'''
b = np.array([0.62, 0.42, -0.28, -1.78, 0.52, 0.49, -1.24, 0.45, 0.40, 0.23, 0.72, -1.20, 0.60, -0.60, 0.33])
b = [ 0.57148148  0.3690404  -0.33592593 -1.82919192  0.46584175  0.43806397 -1.29341751  0.39681818  0.34547138  0.17080808  0.6090404  -1.30762626  0.49414141 -0.71292929  0.21636364]
'''

r_mean = 3.83

train = np.array([
    [1,1,1,0,0],
    [0,1,1,0,1],
    [1,1,0,0,1],
    [0,0,1,1,1],
    [1,0,0,1,1],
    [0,1,0,1,1],
    [1,0,1,1,0],
    [1,0,1,0,1],
    [0,1,1,1,0],
    [0,0,1,1,1]]
    )

# a matrix 생성
a = np.zeros((30, 15))

cnt = 0
for col in range(5):
    for row, t in enumerate(train):
        if t[col] == 1:
            # print(col+1, '열', row+1, '행')

            a[cnt][row] = 1
            a[cnt][10+col] = 1
            cnt += 1

# c vector 생성
c = np.array([5,5,4,3,5,4,3,2,3,2,4,5,3,3,4,5,5,1,4,3,2,4,3,4,3,2,5,5,5,4])
c = c-r_mean

# a1 = lin.pinv(np.dot(a.T, a))
# print(a1.shape)
# b = np.dot(a1, c)
# print(b.shape)

# b vector 구함
print('**** Answer ****')
print(np.dot(lin.pinv(a), c))
