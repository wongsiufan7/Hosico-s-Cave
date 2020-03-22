#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Analysis Library import
import MeCab
import neologdn
import re
import pandas as pd
from functools import reduce


# In[2]:


# Visualization Library import & japanese font change(MACのフォットがちょっと上手くいかない)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['font.sans-serif'] = ['AppleGothic']
mpl.rcParams['font.serif'] = ['AppleGothic']
import seaborn as sns
sns.set_style("darkgrid",{"font.sans-serif":['AppleGothic', 'Arial']})


# In[ ]:


# 1.1 日本語前処理


# In[4]:


# Mecab + Neologd辞書　確認
neologd_tagger = MeCab.Tagger('-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
print(neologd_tagger.parse('庭には二羽鶏がいる'))


# In[5]:


#  形態素解析関数
def word_tokenize(x):
    text = neologd_tagger.parse(x)
    lines = text.split('\n')
    words = []
    for line in lines:
        cols = line.split("\t")
        if cols[0] != 'EOS':
            words.append( cols[0] )
    return words


# In[6]:


#  形態素解析関数(名詞だけ取り出す)
def word_tokenize_non(x):
    text = neologd_tagger.parse(x)
    lines = text.split('\n')
    words = []
    for line in lines[0:len(lines)-2]:
        cols = line.split("\t")
        if cols[3].startswith("名詞"):
            words.append(cols[0])
    return words


# In[7]:


#関数確認
word_list = word_tokenize_non('庭には二羽鶏がいる')
print(word_list)


# In[8]:


#タミフルテキスト（1000件）導入
tamifuru = pd.read_json("/Users/samuel/Desktop/source code/text mining/tamifuru_1904081354.json")


# In[9]:


tamifuru = tamifuru.drop(columns = "attinfo") 


# In[10]:


#テキストを形態素解析し、新しいコラムに置く
tamifuru["word_tokenize"] = tamifuru.apply(lambda row : word_tokenize_non(str(row["comment"])), axis =1)


# In[11]:


tamifuru


# In[12]:


#ストップワード
single = r"^[0-9０-９ぁ-んァ-ン！%％？・　!-/:-@≠\[-`{-~\u3001-\u303F]$"
pair    = r"^[ぁ-ん]{2}$"
numb  =r"^[0-9]+$"


# In[13]:


#ストップワードフィルター函数
def stopword(words):
    return [ x for x in words if len(x) > 1 and re.match( pair, x ) is None and re.match( numb, x ) is None]


# In[14]:


#ストップワードフィルター
tamifuru["word_refined"] = tamifuru.apply(lambda row: stopword( row["word_tokenize"] ), axis=1)


# In[15]:


tamifuru


# In[ ]:


# 1.1 Word Cloud


# In[16]:


#Word Cloud生成
fpath = "/Users/samuel/Library/Fonts/NotoSansCJKjp-Regular.otf"

wordcloud = WordCloud(
    background_color="white",
    max_font_size=80,
    relative_scaling=.4,
    width=900,
    height=500,
    font_path=fpath,
    ).generate(str(tamifuru['word_refined']))
plt.figure(figsize=(15,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:


# 1.2  frequency Bar Plot


# In[17]:


# 単語集計函数
def wordcount(dic, items):
    for item in items:
        if item not in dic:
            dic[item] = 1
        else:
            dic[item]+=1
    return dic


# In[18]:


word_count = reduce( lambda d, tokuns: wordcount(d, tokuns) , list(tamifuru.word_refined), dict())


# In[20]:


#　Dataframeにし、ソートする
word_count_list = pd.DataFrame(list(word_count.items()), columns=['word','count'])
sorted_word_count_list = word_count_list.sort_values('count', ascending=False)


# In[21]:


sorted_word_count_list


# In[22]:


# Bar plot生成
plt.figure(figsize=(15,10))
sns.barplot(x=(sorted_word_count_list['word'])[0:20], y=(sorted_word_count_list['count'])[0:20])
plt.xticks(rotation= 45)
plt.xlabel('単語')
plt.ylabel('頻度')
plt.title('頻度表')


# In[ ]:




