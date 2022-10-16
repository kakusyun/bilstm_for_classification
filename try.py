import numpy as np

def why():
  print('it is a test')

c = np.array([[1, 2, 5, 9, 9, 9, 3]])
d = np.argmax(np.bincount(c.flatten()))

print(d)

print(c[0, 1])

print('I am trying.')
