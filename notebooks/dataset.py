import sys
sys.path.append('..')

from data_generator import raw_data
dataset = raw_data
print(dataset.head())