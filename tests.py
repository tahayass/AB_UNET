from scipy.stats import entropy as ent
import numpy as np
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
import cupy as cp



x_on_gpu0 = cp.array([1, 2, 3, 4, 5])
print(x_on_gpu0.device)




