import json
import math

import numpy as np
import pandas as pd

from recommendation import Recommend

if __name__ == '__main__':
    r = Recommend(needInit=True, n=1000)
    user_sim_matrix_by_rating = np.loadtxt(open("./user_sim_matrix.csv", "rb"), delimiter=",")
    res1 = r.get_CFRRecommend(user_sim_matrix_by_rating, r.user_map[1], 10, 10)
    print(res1[:10])
    res = r.recommend(1)
    print(res[:10])