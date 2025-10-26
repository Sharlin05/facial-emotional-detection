import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from train_model import load_data, DATA_DIR

X, y = load_data(DATA_DIR)
print('Loaded shapes:', X.shape, y.shape)
