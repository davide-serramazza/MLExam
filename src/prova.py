import numpy as np
import pandas as pd
from sklearn.preprocessing import *

df = pd.read_csv("../MLCup/ML-CUP17-TR.csv", comment='#', header=None)
features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
targets_col = ["target_x", "target_y"]
df.columns = ["id"] + features_col + targets_col

print df.describe()

#normalized_df = normalize(df, axis=0)  # normalize each feature between 0 and 1 independently
#normalized_df = pd.DataFrame(normalized_df, columns=df.columns)
#print normalized_df.head()

standardizer = StandardScaler()
standardized_df = standardizer.fit_transform(df)
df = pd.DataFrame(standardized_df, columns=df.columns)
print df.describe()
