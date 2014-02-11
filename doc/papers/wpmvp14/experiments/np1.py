#pythran export np1(int)
import numpy as np
def np1(n):
  x = np.empty(n, np.float32)
  return np.sum(100.*(x[1:] - x[:-1] ** 2) ** 2
                + (1. - x[:-1]) ** 2)
