import pandas as pd

df = pd.read_csv("TaskOne/BA_reviews.csv")


df.reset_index(drop=True, inplace=True)

print(df['reviews'])
df.info()
print(df.describe())
print(df.count())

df['reviews'] = df['reviews'].str.strip()
df['reviews'] = df['reviews'].str.strip('Not Verified |')
df['reviews'] = df['reviews'].str.strip('✅ Trip Verified |')
df['reviews'] = df['reviews'].str.lower()

print(df)

df.to_csv("Clean_Data.csv")