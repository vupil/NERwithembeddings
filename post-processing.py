# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/gdrive')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.manifold import TSNE
import seaborn as sns

all_axioms_path = r'/content/gdrive/MyDrive/Eli Lilly/risk grids/AllAxioms.csv'
all_classes_path = r'/content/gdrive/MyDrive/Eli Lilly/risk grids/AllClasses.csv'

all_classes = pd.read_csv(all_classes_path, names=['classes'])
all_axioms = pd.read_csv(all_axioms_path, names=['class1', 'rel', 'class2'], sep=' ')

all_classes.head()

all_axioms.head()


all_axioms_path = r'/content/gdrive/MyDrive/Eli Lilly/risk grids/VecResults.csv'
vec_file_path = r'/content/gdrive/MyDrive/Eli Lilly/risk grids/VecResults.csv'

vec_embeddings_raw = pd.read_csv(vec_file_path, header=None)

vec_embeddings_raw.head()

embeddings_data = vec_embeddings_raw.iloc[:,1:].values.T
# remove special characters ('[',']') from first and last rows of elements
embeddings_data[0] = np.array([re.sub('[^A-Za-z0-9]+', '', x) for x in embeddings_data[0]])
embeddings_data[-1] = np.array([re.sub('[^A-Za-z0-9]+', '', x) for x in embeddings_data[-1]])

embeddings_data = embeddings_data.astype(float)

embeddings_df = pd.DataFrame(data=embeddings_data, columns=vec_embeddings_raw.iloc[:,0])
embeddings_df.shape

embeddings_df.head()



