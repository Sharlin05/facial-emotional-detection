import sys
import os
# ensure project root is on sys.path so train_model can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from train_model import load_data, EMOTIONS, DATA_DIR

train_sub = os.path.join(DATA_DIR, 'train')
if os.path.isdir(train_sub):
    data_dir = train_sub
else:
    data_dir = DATA_DIR

X, y = load_data(data_dir=data_dir)
print('Loaded images:', X.shape)
import numpy as np
counts = np.sum(y, axis=0).astype(int)
for e, c in zip(EMOTIONS, counts):
    print(f'{e}: {c}')
