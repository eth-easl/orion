import pandas as pd
import sys

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])
names1 = list(df1['Name'])
names2 = list(df2['Name'])
assert names1==names2
