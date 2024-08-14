import pickle
import gzip
with gzip.open('D:\\fyp_veido\\phoenix14t.pami0.train.annotations_only.gzip', 'rb') as f:
    annotations = pickle.load(f)
print(annotations)
