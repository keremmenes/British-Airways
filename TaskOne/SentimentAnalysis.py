import pandas as pd

df = pd.read_csv('TaskOne/Clean_Data.csv')

print(df)

df['reviews'] = df['reviews'].str.replace('[^\w\s]','')
print(df['reviews'])