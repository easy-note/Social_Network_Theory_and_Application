import numpy as np

# numpy output on a single line
np.set_printoptions(linewidth=np.inf)

h = np.array(
    [[0, 1/2, 1/2, 0, 0, 0, 0, 0],
    [1/2, 0, 0, 0, 1/2, 0, 0, 0],
    [0, 1/2, 0, 0, 0, 0, 0, 1/2],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1/2, 0, 0, 0, 1/2],
    [0, 0, 0, 1/2, 1/2, 0, 0, 0],
    [0, 0, 0, 1/2, 0, 1/2, 0, 0],
    [1/3, 0, 0, 1/3, 0, 0, 1/3, 0]]
)


theta = 0.85
n = 8
g = theta * h + (1-theta) * 1/n * np.ones((8, 8))

ans_pi = np.full(8, 1/8)
print('0 iter : ', ans_pi)

for i in range(20):
    ans_pi = ans_pi.dot(g)
    print(f'{i+1} iter : ', ans_pi)

print('\nHigh importance node oder : {}'.format(ans_pi.argsort()[::-1]+1))
