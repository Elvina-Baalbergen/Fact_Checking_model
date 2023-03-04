import numpy as np
import pandas as pd

metadata = pd.read_csv('./data/raw/MovieSummaries/movie.metadata.tsv', delimiter='\t')
print(metadata.head(5))
