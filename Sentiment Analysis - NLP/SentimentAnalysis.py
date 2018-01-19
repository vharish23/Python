%matplotlib inline
import pandas as pd
import nltk 
import re 
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.util import ngrams
from pylab import *


def process_reviews( raw_review ):
    
    review_text = BeautifulSoup(raw_review).get_text()
    
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    
    words = letters_only.lower().split()
    
    stops = set(stopwords.words("english"))
    
    meaningful_words = [w for w in words if not w in stops] 
    
    return( " ".join( meaningful_words ))

def get_ngrams(df_temp):
    data = df_temp["Processed_Text"].str.cat(sep=' ')
    n = 2
    bigrams = ngrams(data.split(), n)
    bigstr =""
    for grams in bigrams:
        str1 = grams[0]+"_"+grams[1]
        bigstr += " "+str1
    return bigstr

def get_wordcloud(bigstr):
    wordcloud = WordCloud(width=800, height=400).generate(bigstr)
    return wordcloud
  
def plot_pi_chart(pid):
    rat_count_list =[]

    for i in range(1,6):
        rat_count_list.append(len(df[df['ProductId'].isin([pid])].loc[df["Score"] == i].index)) 

    figure(1, figsize=(12,12))
    ax = axes([0.1, 0.1, 0.8, 0.8])

    labels = '1', '2', '3', '4','5'
    explode=(0, 0, 0, 0,0)

    pie(rat_count_list, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)

def plot_review_polarity(pol_pid):
    ratting_polarity_list=[]
    for ratting_senti in range(1,6):
        dummy = []
        de_s = df[df['ProductId'].isin([pol_pid])].loc[df["Score"] == ratting_senti]
        dummy.append(TextBlob(de_s["Text"].apply(process_reviews).str.cat(sep=' ')).sentiment.polarity)
        ratting_polarity_list.append(dummy)
    polarity_bar_list = pd.DataFrame(ratting_polarity_list,
                                 index=['rating 1', 'rating 2', 'rating 3', 'rating 4','rating 5'],columns=['values'])
    polarity_bar_list.sort().plot(kind='barh' )

def analyze_reviews(prodNo):
    
    df = pd.read_csv('D:/Study material - Course work/Spring 2016/IDS 566 - Text Analytics/Project/amazon-fine-foods/Reviews.csv')
    
    plot_pi_chart(prodNo)
    
    plot_review_polarity(prodNo)

    prod_list_sat=[]
    wordCloud_string_list =[]
    prod_list_sat.append(prodNo)
    
    df_sat_prd = df[df['ProductId'].isin(prod_list_sat)].loc[df["Score"] >3]
    df_sat_prd["Processed_Text"]=df_sat_prd["Text"].apply(process_reviews)
    bigstr_sat =get_ngrams(df_sat_prd)
    wordCloud_string_list.append(bigstr_sat)
   
    df_unsat_prd =df[df['ProductId'].isin(prod_list_sat)].loc[df["Score"]<2 ]
    df_unsat_prd["Processed_Text"]=df_unsat_prd["Text"].apply(process_reviews)
    bigstr_unsat=get_ngrams(df_unsat_prd)
    wordCloud_string_list.append(bigstr_unsat)
    
    
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111)
    for i in range(0,len(wordCloud_string_list)):
        ax = fig.add_subplot(2,1,i+1)
        wordcloud = get_wordcloud(wordCloud_string_list[i])
        ax.imshow(wordcloud)
        ax.axis('off')
        if i==0:
             ax.set_title('Analysis of Positive Reviews' ,fontsize=30)
        else:
            ax.set_title('Analysis of Negative Reviews',fontsize=30)
   
    ax.axis('off')
    fig.subplots_adjust(hspace=2)
    plt.tight_layout(pad=0)
    plt.show()
