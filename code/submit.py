import numpy as np
import pandas as pd
from test import *


global answer_path
answer_file_path = './answer.csv'

answer_dict = np.load(answer_path, allow_pickle=True)['arr_0']
answer_dict = answer_dict.item()

length = len(answer_dict)

answer_file = []

for idx in range(length):
	answer_file.append((idx, answer_dict[idx]))

answer_file = pd.DataFrame(answer_file, columns=["Id", "Predicted"])
answer_file.to_csv(answer_file_path, index=False)
