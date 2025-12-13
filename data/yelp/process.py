import pickle
from scipy.sparse import coo_matrix
import numpy as np

def txt_to_coo(path):
    rows = []
    cols = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            if len(values) < 2:
                continue

            src = int(values[0])
            dsts = map(int, values[1:])

            rows.extend([src] * (len(values) - 1))
            cols.extend(dsts)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.ones(len(rows), dtype=np.int8)

    return coo_matrix((data, (rows, cols)))

train_coo = txt_to_coo("train.txt")
with open("trnMat.pkl", "wb") as f:
    pickle.dump(train_coo, f)

test_coo = txt_to_coo("test.txt")
with open("tstMat.pkl", "wb") as f:
    pickle.dump(test_coo, f)

