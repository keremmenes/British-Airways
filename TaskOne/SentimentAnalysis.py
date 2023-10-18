import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('TaskOne/Clean_Data.csv')

print(df)

df['reviews'] = df['reviews'].str.replace('[^\w\s]','')

print(df['reviews'])

print(df.iloc[1,1])
df['reviews'] = df.apply(lambda row: nltk.word_tokenize(row['reviews']), axis=1)
print(df.iloc[0,1])

df['reviews'] = df['reviews'].apply(lambda x: ' '.join([word for word in x if word not in (stop_words)]))

print(df.head(20))

def polarity_calc(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return None
    
def tag_cal(num):
    if num<0:
        return 'Negative'
    elif num>0:
        return 'Positive'
    else:
        return 'Neutral'
        
    
df['polarity'] = df['reviews'].apply(polarity_calc)


df['tag'] = df['polarity'].apply(tag_cal)


print(df)

print((df.groupby('tag').size()/df['tag'].count())*100)




"""                 VISUALIZING                  """



text = " "
for ind in df.index:
    if df['tag'][ind] == "Positive":
        text = text + df['reviews'][ind]
      
wordcloud_positive = WordCloud().generate(text)

plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Words')
plt.axis("off")
plt.show()

text2= " "        
for ind in df.index:
    if df['tag'][ind] == "Negative":
        text2 = text2 + df['reviews'][ind]  
wordcloud_negative = WordCloud().generate(text2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Words')
plt.axis("off")
plt.show()

df['tag'].value_counts().plot(kind='bar')
sns.set(font_scale=1.4)
df['tag'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Sentiment", labelpad=14)
plt.ylabel("No of reviews", labelpad=14)
plt.title("Bar Chart", y=1.02)

print(df['tag'])

tag_counts = df['tag'].value_counts()


plt.figure(figsize=(6, 6))
plt.pie(tag_counts, labels=tag_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart')
plt.axis('equal')
plt.show()

tag_counts.plot(kind='barh')
plt.title('Horizontal Bar Chart')
plt.xlabel('Numbers')
plt.ylabel('Tags')
plt.show()
