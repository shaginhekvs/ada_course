
# coding: utf-8

# # HELPER FUNCTIONS
#
# In this notebook, we define the set of helper functions that we use to help in the analysis. Given the length of the notebook, we have decided to separate these functions from the analysis itself to make it more readable.
#
# The rest of this notebook is organized as follows. We have divided it in sections, depending on what the functions fo. Thus, we have::
# * [Tweets processing and reading](#processing): for processing the tweets read from the database and the APIs and also defining the preprocessing functions used internally.
# * [Basic analysis](#basic): functions for plotting number of tweets and retweets
# * [Language](#language): analysis of the different languages of the tweets
# * [Topics modelling](#topics): function for obtaining the different topics discussed on the tweets so __we can observe how they tend to change over time and what is discussed__
# * [Hashtag analysis](#hashtag): different functions for creating wordclouds of hashtags and studying coocurrence of hashtags theoretically confronted
# * [Sentiment analysis](#sentiment): functions performing sentiment analysis of the tweets that allow us to study the polarity and subjectivity
#

# In[26]:

#Imports
import re
import pandas as pd
import string
import pickle
import os
import codecs
import io
import numpy as np
from datetime import datetime
import datetime as dt
import json
import time
import numpy
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import random
from functools import reduce
import pyLDAvis.gensim

#Error message if an import fails
some_failed=False

try:
    from textblob import TextBlob
except:
    get_ipython().system('conda install -y -v -c conda-forge textblob')
    some_failed=True
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import TweetTokenizer
except:
    get_ipython().system('conda install -y -v -c conda-forge nltk')
    some_failed=True
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaMulticore
except:
    get_ipython().system('conda install -y -v -c anaconda gensim')
    some_failed=True
try:
    from googletrans import Translator
except:
    get_ipython().system('pip install googletrans')
    some_failed=True
try:
    import tweepy
    from tweepy import OAuthHandler
except:
    get_ipython().system('conda install -y -v -c conda-forge tweepy ')
    some_failed=True
try:
    from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator
except:
    get_ipython().system('conda install -y -v -c https://conda.anaconda.org/amueller wordcloud')
    some_failed=True

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api
try:
    import plotly.plotly as py
    import plotly.graph_objs as go
except:
    get_ipython().system('pip install plotly')
    some_failed=True

try:
    from sqlalchemy import create_engine
except:
    get_ipython().system('pip install sqlalchemy')
    some_failed=True

if(some_failed):
    print('restarted kernel with dependencies installed, please run cell again')
    os._exit(00)

get_ipython().magic('matplotlib inline')


# In[35]:

#Constants
class style:
   BOLD = '\033[1m'
   END = '\033[0m'


# <a id='processing'></a>
# ## Tweets processing and reading
#
# In this section we define a series of functions used for reading tweets from the dataset, the Tweepy API and their basic processing.
#
# #### Database
#
#  _Format of the tweets_
#
# The raw tweets obtained from the database have the following format:
#
# | 2 characters | 18 characters   | weekday month day  HH:MM:SS time_zone year| user | *optional* | text
# |:----:|:----:|:----:|:----:|:----:|:----:
# |   language id  | tweet id | date | username | (RT) | text
#
#
# The first two characters indicate the language of the tweet. Next is the 64 bit (18 characters) ID of the tweet. In the next field we can find the date, showing first weekday, followed by month, day of the month, hour , the time zone and finally the year. We can then find the tag *RT* if is a retweet. Finally, we have the tweet itself.
#
# _Reading_
#
# We create a function *read_tweet_files* for reading the tweets filtered by hashtag from the database. This function takes the hashtag used for filtering (to identify the file) and returns a list with one tweet per line.

# In[2]:

def read_tweet_files(hashtag):
    '''
    Read tweets from the txt file gotten from cluster
    INPUT:
        hashtag: hashtags for which we have data
    OUTPUT:
        tweets_list,hashtag:list of tweets in file, hashtag searched for
    '''

    tweets_list=[]
    #path
    file_name='../tweets_datasets_cluster/tweets'
    #name of file depending on the hashtag
    if hashtag=='#blacklivesmatter':
        file_name+='_blm'
    elif hashtag=='#alllivesmatter':
        file_name+='_against_blm'
    else:
        return [],hashtag
    file_name+='.txt'

    #read file
    with io.open(file_name,'r',encoding='utf-8') as file:
        for line in file:
            if(':') in line:
                line=line.replace('\\t','\t')
                tweet_text=line.split(':',maxsplit=1)[1][:-2]
                tweets_list.append(tweet_text)
    return tweets_list,hashtag


# _Processing_
#
# To process these tweets in raw format we create the function *process_tweets* which given the list of tweets obtain from the MapReduce jobs, returns a dataframe which the tweets divided by their fields, that is with the columns:
#    * lang: language of the tweet
#    * id: id of the tweet, converted to int
#    * date: with the format year/month/day
#    * epoch: timestamp aggregated by day
#    * user: who tweeted
#    * text: of the tweet
#
# In addition, we also add a column with the hashtags found in the tweets, the hashtag used in the filtering, and if it's a retweet the handle of the user who first tweeted:
#    * hashes_inside: hashtags found in the tweets
#    * hashtag: hashtag as filter to obtain the tweet
#    * to: retweet handles inside the tweet

# In[28]:


def process_tweets(tweet_list,hashtag):
    '''
    Process tweets from the database dividing them by their fields
    INPUT:
        tweet_list: tweet as returned by the databasee
    OUTPUT:
        dataframe with one row per tweet and columns
            -lang: language of the tweet
            -id: id of the tweet
            -date: with the format year/month/day
            -epoch: timestamp of the tweet aggregated by day
            -user: who tweeted
            -text: of the tweet
            -hashtag:which tweet definitely contains
            -hashes_inside: hashtags in the text
            -to:retweet handles inside tweet
    '''

    #Dictionary for storing intermediate values
    d={'lang':[], 'id':[],'date':[], 'epoch':[], 'user':[], 'text':[],'hashtag':[],'hashes_inside':[],'to':[]}
    #Iterate through every tweet in the list
    for i,tweet in enumerate(tweet_list):
        #Initalize values
        lang,id_,day,month,year,epoch,user,to,hashes_in,text,dt='en','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1'
        try:
            #Split them by spaces
            token=tweet.split()

            #Text: join the tokenized words separating them with spaces
            text=" ".join(token[9:])
            #Language
            lang=token[0]
            #Tweet id
            id_=int(token[1])
            #Date
            day=token[4]
            month=token[3]
            year=token[7]
            dt = datetime.strptime(year+' '+month+' '+day, '%Y %b %d')
            epoch=time.mktime(dt.timetuple())
            #user who tweeted
            user=token[8]
            #Language
            lang=token[0]


            #Hashtags and user mentions
            hashes_in='' #hashtags in text
            to='' #user mentions

            #for each word in the text
            for token in token[9:]:
                #If it's a hasthag we add it to the list converting it to lowercase
                if token[0]=='#':
                    hashes_in=hashes_in+' '+token.lower()
                #If it's a retweet, get the handles of the user who tweeted it
                elif token[0]=='@':
                    to_add=token
                    if(to_add[-1])==':':
                        to_add=to_add[:-1]
                    to=to+' '+to_add

        #If missing field, ignore tweet
        except:
            continue

        #Language
        d['lang'].append(lang[1:])
        #ID
        d['id'].append(id_)

        #Date
        d['epoch'].append(epoch)
        d['date'].append(dt)
        #Username
        d['user'].append(user)
        #Text
        d['text'].append(text)
        d['hashtag'].append(hashtag)
        d['hashes_inside'].append(hashes_in)
        d['to'].append(to)
    #Create dataframe from the dictionary and set the id as the index
    df=pd.DataFrame.from_dict(d).set_index('id')
    df['date']=pd.to_datetime(df['date'])
    return df



# #### Twitter API
#
# In addition to the database, we also use the Twitter API to obtain data, as the database does not  have recent data and is missing information such as the number of retweets. To scrape the data we use the GetOldTweets-python API ( https://github.com/Jefferson-Henrique/GetOldTweets-python) saving the data in a sqlite database.
#
# We read the data using the read_tweets.py file (which can be found here) which saves them to the database. In addition, we also define the function *read_database* which reads each of the tables (one per protest) found in the SQLite database into a dataframe.
#
# We process the result with *conform_dataframes*, which process the dataframe received by the previous function to the same format as the rest of dataframes used.

# In[29]:


def read_database(table,engine=create_engine('sqlite:///../data.db')):
    '''
        Read the downloaded tweets from twitter saved in the database.
        Table Names - Hashtags
        INPUT:
            table: table in which we have data
        OUTPUT:
            df: dataframe of the data from the table
        1. blm - #blacklivesmatter
        2. maga - #maga, #makeamericagreatagain
        3. antimaga - #NotMyPresident , #TheResistance , #Resist , #TakeAKnee
        4. misogyny -  #MeToo, #WHYWEMARCH,  #YesAllWomen,  #EverydaySexism, #WhyIStayed, #NotOkay, #VOTEPROCHOICE
        5. bring_back - #bringbackourgirls, #ChibokGirls, #bbog
        6. antblm  - #BlueLivesMatter, #AllLivesMatter, #WhiteLivesMattter, #NYPDLivesMatter,  #PoliceLivesMatter
    '''
    df=pd.read_sql_query('SELECT * FROM {}'.format(table),engine,parse_dates=True,index_col='id')
    df.drop('geo',axis=1,inplace=True)
    return df

def conform_dataframes(dest,hashtag,cols_mapping=None):
    '''
        Read the downloaded tweets from twitter.
        Table Names - Hashtags
        INPUT:
            table: table in which we have data
        OUTPUT:
            df: dataframe of the data from the table
        1. blm - #blacklivesmatter
        2. maga - #maga, #makeamericagreatagain
        3. antimaga - #NotMyPresident , #TheResistance , #Resist , #TakeAKnee
        4. misogyny -  #MeToo, #WHYWEMARCH,  #YesAllWomen,  #EverydaySexism, #WhyIStayed, #NotOkay, #VOTEPROCHOICE
        5. bring_back - #bringbackourgirls, #ChibokGirls, #bbog
        6. antblm  - #BlueLivesMatter, #AllLivesMatter, #WhiteLivesMattter, #NYPDLivesMatter,  #PoliceLivesMatter
    '''
    if cols_mapping==None:
        cols_mapping={'mentions':'to','username':'user','hashtags':'hashes_inside'}
        dest['hashtag']=hashtag
        dest['epoch']=pd.to_datetime(dest['date']).apply(lambda x:time.mktime(x.to_pydatetime().timetuple()))
    dest.rename(index=str, columns=cols_mapping,inplace=True)



# We also define the *load_dataframes* function, for the #BlackLivesMatter protest. As in this case, we have data from both the Twitter dataset from the cluster and the twitter API (for the 2016 and 2017 data), due to the more recent tweets not being present in the dataset present in the cluster. This function merges the output from both inputs into a single dataframe.

# In[ ]:

def load_dataframes(hashtag):
    '''
        Read the blm and alllm tweets from database and csv.

        INPUT:
            hashtag - Hashtag to read : possibly #blm or #alllm
        OUTPUT:
            df: dataframe of the data from the table and the csv
    '''
    to_do=False
    table=''
    tweet_str=''
    if(hashtag=='#blm'):
        tweet_str='#blacklivesmatter'
        table='blm'
        to_do=True
    elif (hashtag=='#alllm'):
        tweet_str='#alllivesmatter'
        table='antblm'
        to_do=True
    else:
        print('nothing to do')
        to_do=False
    if(to_do):
        tweets_csv=process_tweets(*read_tweet_files(tweet_str))
        df=read_database(table)
        conform_dataframes(df,tweet_str)
        df=tweets_csv.append(df)
        df.drop_duplicates(inplace=True)
        df['date']=pd.to_datetime(df.date)
        return df
    else:
        return pd.DataFrame()



# #### Tweepy API
#
# To obtain the missing values (e.g., number of retweets) from the cluster dataset we use the Tweepy API, passing it the indices of the tweets.
#
# We first define a function for loading the API, were you have to add the credentials.

# In[ ]:

# enter the credentials to access the twitter api
def load_api():

    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_secret = ''
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # load the twitter API via tweepy
    return tweepy.API(auth)


# For the search of these columns we define the function *scrape_missing_columns*, passing it a list of indices to read. Since the Twitter API has some limitations of queries for each user per 15 min interval (explained at length on  https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object), we split the task of gathering additional info about the tweets, indicating with start and end which are indices to be read in this batch.

# In[33]:

def write_tweets(tweets, filename):
    ''' Function that appends tweets to a file. '''

    with open(filename, 'a') as f:
        for tweet in tweets:
            json.dump(tweet._json, f)
            f.write('\n')

def scrape_missing_columns(list_of_ind, start,end):
    '''
    Read from the Tweepy API the missing fields
    INPUT:
        list_of_ind: list of indices to search for
        start: first index to read
        end: last index to read

    '''
    list_of_ind=pd.DataFrame(list_of_ind)
    # we wait 15 minutes if the a tweepy.TweepError which should be due to the limit of queries exceeded per user.

    # one batch is allowed to gave 100 twitters / query
    # MODIFY ACCORDINGLY idx_by_100 for every app
    idx_by_100 = list(range(start, end, 100))
    # distribution per single user, use 3 apps
    # for each app define idx_by_100 to be data_split[0]. For consistent file naming, use the intervals such that their
    # beginning value is divisible by 100 and their ending value ends in 999 or is the end (see above intervals)
    data_split = np.linspace(start,end, 4)
    # For each batch of 1000 tweets, we write them to a json file.
    # we wait 15 minutes if the a tweepy.TweepError which should be due to the limit of queries exceeded per user.
    last_tweet = idx_by_100[-1]
    for count, idx_value in enumerate(idx_by_100):
        last_tweet_batch = min((idx_value+100), last_tweet)
        print(idx_value)
        range_batch = list_of_ind[idx_value:last_tweet_batch]['id'].tolist()
        filename = 'tweets_app'

        try:
            tweets_batch = api.statuses_lookup(range_batch)
            write_tweets(tweets_batch, filename)
            print('tweets recovered: ', len(tweets_batch), ' filename ', str(int(idx_value/1000) ))
            print('progress: ', idx_value, ' out of ', last_tweet)
        except tweepy.TweepError as e:
            print(e)
            print('exception raised, waiting 15 minutes')
            print('(until:', dt.datetime.now()+dt.timedelta(minutes=15), ')')
            time.sleep(15*60)


# For reading back in the tweets, we design the following procedure.
#
# i) we start with a dataframe with a single column representing the index of the tweets whose hashtag we are looking at, and that we obtained from the cluster. That column is the index of the dataframe.
#
# ii) we search through the json files collected from the api and we sequentially complete the relevant fields we are interested in in the dataframe. We dont read more than a fixed number of tweets at once because they eat up a lot of memory and we are only interested in a few fields. So we discard the tweets every once in a while.
#
# As the location of tweets is too sporadic, we are only interested in the number of retweets

# In[6]:

def read_tweepy_result(df, hashtag):
    '''
    Reads the json files obtain with Tweepy
    INPUT:
        df: original dataframe which id's we searched for
        hashtag: hashtag we are searching for
    OUTPUT:
        dataframe with the columns read from the API added
    '''

    #Initialize the id with the index values of the original dataframe
    result = pd.DataFrame({'id': df.index.values})
    result = result.set_index('id')
    # create a dictionary that maps attributes of interest to ways of accessing them in a twitter object.
    attrib_of_interest = {'tweet_location': ['place', 'name'],                           'user_location': ['user','location'], 'user_followers':['user','followers_count'],                          'retweets_No': ['retweet_count']}
    attr_types = {'tweet_location': 'str', 'user_location': 'str', 'user_followers' : 'float64', 'retweets_No':'float64' }

    #Read the tweets
    result =read_twitter_json_all(hashtag, result, attrib_of_interest, attr_types)

    #Replace missing values
    result['tweet_location'].replace('nan', inplace = True)
    result['user_location'].replace('nan', inplace = True)
    #Date column
    result['date'] = df['date']

# function that given a list of tweets and a list of attributes of interest, will fill the dataframe
def fill_in_tweets(attributes_of_interest, raw_tweets_api, df_id_api):
    for raw_tweet in raw_tweets_api:
        for k,v in attrib_of_interest.items():
            #initialisation
            if (v):
                attribute_value = raw_tweet[v[0]]
            idx = 1
            while (idx < len(v) and attribute_value):
                attribute_value = attribute_value[v[idx]]
                idx = idx +1
            if(attribute_value):
                df_id_api.at[raw_tweet['id'], k] = attribute_value
            else:
                df_id_api.at[raw_tweet['id'], k] = np.nan
            #print(attribute_value)
    return df_id_api

# function that reads all the twitters in one json file and places them in a list
def read_twitter_json(filename, df_id_api, attributes_of_interest):
    tweets_api = []
    with open(filename, 'r') as f:
        for i,line in enumerate(f.readlines()):
            tweets_api.append(json.loads(line))
            if len(tweets_api) > 100:
                df_id_api = fill_in_tweets(attributes_of_interest, tweets_api, df_id_api)
                tweets_api = []   # empty the list, read other tweets.

    return df_id_api


# function that locates all the relevant files which include the twitters searched fo
# and initialises the columns of interest to deafault value NAN
def read_twitter_json_all(hashtag, df_id_api, attributes, atypes):
    condition=False
    df_id_api = df_id_api.reindex(columns = list(attributes.keys()) )
    df_id_api = df_id_api.astype(atypes)
    files_read = 0
    for filename in os.listdir('./k/'):
        if(hashtag=='#blacklivesmatter'):
            condition=('tweets_app' in filename) and ('all' not in filename)
        elif(hashtag=='#alllivesmatter'):
            condition='tweets_app' in filename and 'all' in filename
        if(condition):
            #print('reading {}'.format(filename))
            df_id_api = read_twitter_json('./k/' + filename, df_id_api,attributes )
            files_read = files_read + 1
    return df_id_api


# #### Preprocessing
#
# Prior to the topics modelling and to other analysis, tweets have to be preprocessed. We use several tecniques for such preprocessing:
# * Remove slang: We replace slang words and diminutives frequently used in twitter with normal words. In order to do so, we obtain a slang dictionary by scraping https://www.noslang.com/dictionary/, which we use to replace do the replacement by calling the function replace slang. The webscraping can be found in the file *scrape_slang_dictionary.ipynb*
# * Remove handles: Remove @user and mentions
# * Reduce the length: We reduce the length of words such as tweeeeet for tweeet, so all words with more than three letters repeated get reduced to three letters and count as one word.
# * Remove stopwords: We use a dictionary of stopwords from nltk to remove all stopwords, which are not informative of the topic.
# * Remove punctuation: We remove all punctuation from the tweets
# * Remove urls: We remove all URLs from the tweets.
# * Remove RT tag: We remove the RT tag of tweets, as it should not be considered a topic.
# * Remove numbers: We remove all numbers from the tweets, as they are not informative of the topic.
#
# This preprocessing is implemented in the function *preprocess_tweet*, which given a tweet, first tokenizes it and then applies all the tecniques that are set to *True* of the above mentioned. It returns the tweet tokenized. All tweets are converted to lowercase so there is not a difference in words due to capitalization (for example, #BlackLivesMatter is the same as #blacklivesmatter)

# In[34]:

def replace_slang(token, slang):
    '''
    Replace slangs words with dictionary
    INPUT:
        token: word to replace
        slang: slang dictionary
    OUTPUT:
        replaced word
    '''
    #if token in slang dictionary, replace it
    try:
        return slang[token]
    #token not in slang dictionary
    except:
        return token


def preprocess_tweet(tweet, handles=True, length=True, stop=True, punctuation=True,
                     numbers=True, urls=True, retweet=True, slang=True, path_slang='./'):
    '''
    Preprocess tweet with different possibilities.
    INPUT:
        tweet: string with the tweet
        handles: if true, remove handles (user mentions)
        length: if true, reduce length of more than three characters repeated to three characters, e.g, cooool -> coool
        stop: if true, remove stopwords
        punctuation: if true, remove punctuation
        numbers: if true, if true, remove numbers
        urls: if true, remove urls
        retweet: if true, delete RT tag.
        slang: if true, replace slang words
        path_slang: path to the pickled slang dictionary
    OUTPUT:
        list with tokenized processed tweet
    '''

    #Convert tweet to tokens, remove handles and reduce length. Convert tweet to lowercase
    tknzr = TweetTokenizer(strip_handles=handles, reduce_len=length)
    tweet=tknzr.tokenize(str(tweet).lower())

    #Replace stopwords
    if stop:
        #List of stopwords
        stop = stopwords.words('english')
        #add https of links and rt to stopwords
        stop.append('https') #add https
        stop.append('RT') #add RT
        #Remove elemnents that are in list of stopwords
        tweet = [token for token in tweet if token not in stop]

    #Remove punctuation
    if punctuation:
        #List of punctuations
        exclude = set(string.punctuation)
        #Remove elemnents that are in list of punctuations
        tweet = [token for token in tweet if token not in exclude]

    #Remove numbers
    if numbers:
        tweet = [token for token in tweet if not (token.isdigit()
                                         or token[0] == '-' and token[1:].isdigit())]

    #Remove URLS
    if urls:
        #With regexp
        tweet=[re.sub(r'http\S+', '', token) for token in tweet]

    #Remove retweet tag
    if retweet:
        tweet=[token.replace('RT:', '') for token in tweet]

    #Replace slang
    if slang:
        #Load slang dictionary
        slang_dict=pickle.load(open(path_slang+'slang_dict.pkl','rb'))
        #replace slang
        tweet=[replace_slang(token, slang_dict) for token in tweet]

    return tweet


# <a id='basic'></a>
# ## Basic analysis
#
# This section contains functions for plotting the number of tweets and retweets of a protest. They are two types of plots: static, which can be executed and shown inline in the notebook; and interactive, which are linked to plotly and are not shown inline.
#
# The first two function we define are general for all cases.
# * *date_restrictions*: given a dataframe and a date, removes all data from a dataframe prior to that given date.
# * *plot_events*: given a set of events (date and description), this function adds markers to a plot to indicate the most remarkable events
#
#

# In[9]:

def date_restrictions(data, date_to):
    '''
    Ignore data previous to date_to
    INPUT:
        data: dataframe with column 'date'
        date_to: string with format YYYY-MM-DD, ignore data previous to that date
    OUTPUT:
        data filtered
    '''
    #If date_to is not empty, restrict date from it
    if date_to:
        data=data[data['date']> date_to]

    return data


# In[10]:

def plot_events(events, data_grouped, offset=300):
    '''
    Plot events on top of existing plot

    INPUT:
        events: dataframe with events. One column with dates named 'date, with format 'YYYY-MM-DD'
                and one column with the event description, named 'event'
        data_grouped: data grouped to plot on top the events. index should be the date
        offset: of the numbers above the circles

    '''
    #Set column date as index
    events=events.set_index('date')

    #Get the values in the dates or zero if there's none
    data_events=data_grouped.get(events.index,0)

    #Plot circles on the dates and with the height of the values obtained
    plt.plot(events.index, data_events.loc[events.index].fillna(0),'o', markeredgecolor='k',
                                                              markerfacecolor='None',
                                                              markersize=10)

    #Add a number and description to each event
    for n, coor in enumerate(zip(events.index, data_events)):
        plt.text(coor[0], coor[1]+offset, str(n), color="k",clip_on=False,fontsize=14)
        print(n, events.loc[coor[0]]['event'])



# #### Number of tweets
#
# These functions plot the number of tweets and retweets aggregated by day. They take a principal dataframe with its correspondent label, and optionally they can take a second dataframe and label (theoretically, of the protest against) and series of events to plot on top. If a date is given, only data starting from it onwards is plotted, that is, all previous data is ignored.

# In[11]:

def plot_num_tweets(data1, label1, data2=pd.DataFrame(), label2='', events=pd.DataFrame(), date_to=''):
    '''
    Plot of the number of tweets over time, grouped by day
    INPUT:
        data1: principal data to plot.
        label1: label of data1
        data2: secondary data (tweets against protest)
        label2: label of data2
        events: dataframe with the events
        date_to: string with format YYYY-MM-DD, ignore data previous to that date
    '''
    #Process data 1
    #Ignore dates before date_to
    data1['date']=data1.copy().date.dt.to_period("d") #Aggregate by day
    #Ignore dates before date_to
    data1=date_restrictions(data1, date_to)
    #Group it by day and count the number

    data1_grouped=data1.groupby('date').size()

    #Plot data
    ax=data1_grouped.plot(kind='line',
                         figsize=[15,6],
                         label=label1,
                         legend=True)

    #Process data2
    if not data2.empty:
        data1['date']=data1.copy().date.dt.to_period("d") #Aggregate by day
        #Group it by day and count the number
        data2=date_restrictions(data2, date_to)
        #Ignore dates before date_to
        data2_grouped=data2.groupby('date_day').size()
        #Plot data
        data2_grouped.plot(ax=ax,
                          kind='line',
                          label=label2,
                          legend=True,
                          style='r',
                          )
    #Process events
    if not events.empty:
        #Ignore dates before date_to
        events=date_restrictions(events, date_to)
        #Plot events on top of existing plot
        plot_events(events,data1_grouped)

    #Labels
    ax.set_title('Number of tweets of {} protest over time'.format(label1),fontsize=14, fontweight='bold') #Title
    ax.set_xlabel('Month',fontsize=16, fontweight='bold') #xlabel
    ax.set_ylabel('Frequency',fontsize=16, fontweight='bold') #ylabel
    #Change ticks parameters
    ax.tick_params(labelsize=10) #Size
    ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(1), interval=3)) #Position
    ax.xaxis.set_major_formatter(dates.DateFormatter('%d\n%b\n%Y')) #Format



# This functions is an interactive alternative to the previous one, for the data story.

# In[12]:

# alternative function to make it interactive
def plot_num_tweets_interactive(data1, label1, data2=pd.DataFrame(), label2='', events=pd.DataFrame(), date_to=''):
    '''
    Plot of the number of tweets over time (interactive), grouped by day
    INPUT:
        data1: principal data to plot.
        label1: label of data1
        data2: secondary data (tweets against protest)
        label2: label of data2
        events: dataframe with the events
        date_to: string with format YYYY-MM-DD, ignore data previous to that date
    '''
    #Process data 1
    #Ignore dates before date_to
    data1=date_restrictions(data1, date_to)
    #Group it by day and count the number
    data1_grouped=pd.DataFrame(data1.groupby('date').size()).reset_index()
    #Rename columns
    data1_grouped.columns = ['Date', 'TweetsNumber']

    #Trace for principal data
    trace1 = go.Scatter(
        x=data1_grouped['Date'],
        y=data1_grouped['TweetsNumber'],
        name=label1
    )

    #Add to data list
    data=[trace1]

    #Process data2
    if not data2.empty:
        #Group it by day and count the number
        data2=date_restrictions(data2, date_to)
        #Ignore dates before date_to
        data2_grouped=pd.DataFrame(data2.groupby('date').size()).reset_index()
        data2_grouped.columns = ['Date', 'TweetsNumber']

        #Plot
        trace2 = go.Scatter(
            x=data2_grouped['Date'],
            y=data2_grouped['TweetsNumber'],
            name=label2
        )

        #Add to data list
        data.append(trace2)

    #Process events
    if not events.empty:
        trace3 = go.Scatter(
            x=list(events['date']),
            y=np.ones(len(events)) * 5,
            mode='markers',
            text=list(events['event']),
            name='events',
            marker = dict(
                size = 10,
                line = dict(
                    width = 2,
                )
            )
        )
        #adding the events
        data.append(trace3)

    #Plots
    layout = go.Layout(
        showlegend=True,
        title = 'Number of tweets of {} and {} over time'.format(label1,label2)
    )
    fig = go.Figure(data=data, layout= layout)
    plot_url = py.plot(fig, filename='ntweets'+label1)


# #### Number of retweets
#
# These functions have the same signature as the previous ones, with the only difference that in this case we plot the number of retweets grouped by day

# In[22]:

def plot_num_retweets(data1, label1, data2=pd.DataFrame(), label2='', events=pd.DataFrame(), date_to=''):

    #Process data 1

    #Limit the dates
    data1=date_restrictions(data1, date_to)
    #Group it by day and sum the number
    data1_grouped=data1.groupby('date').retweets.agg('sum')

    #Plot
    ax=data1_grouped.plot(kind='line',
                         figsize=[15,6],
                         label=label1,
                         legend=True,
                         marker='|')

    #If they pass us a second datafrmae
    if not data2.empty:
        #Limit the dates
        data2=date_restrictions(data2, date_to)
         #Group it by day and count the number
        data2_grouped=data2.groupby('date').retweets.agg('sum')

        data2_grouped.plot(ax=ax,
                          kind='line',
                          label=label2,
                          legend=True,
                          style='r',
                          marker='|'
                          )

    if not events.empty:
        #Limit the dates
        events=date_restrictions(events, date_to)
        #Plot the events
        plot_events(events,data1_grouped, offset=0)

    #Labels
    ax.set_title('Number of retweets of {} protest over time'.format(label1),fontsize=14, fontweight='bold') #Title
    ax.set_xlabel('Date',fontsize=16, fontweight='bold') #xlabel
    ax.set_ylabel('Frequency',fontsize=16, fontweight='bold') #ylabel
    #Change ticks parameters
    ax.tick_params(labelsize=10) #Size
    ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(1), interval=3)) #Position
    ax.xaxis.set_major_formatter(dates.DateFormatter('%d\n%b\n%Y')) #Format



# In[23]:

# alternative function to make it interactive
def plot_num_retweets_interactive(data1, label1, data2=pd.DataFrame(), label2='', events=pd.DataFrame(), date_to=''):
    '''
    Plot of the number of retweets over time (interactive), grouped by day
    INPUT:
        data1: principal data to plot.
        label1: label of data1
        data2: secondary data (tweets against protest)
        label2: label of data2
        events: dataframe with the events
        date_to: string with format YYYY-MM-DD, ignore data previous to that date
    '''
    #Process data 1
    #Ignore dates before date_to
    data1=date_restrictions(data1, date_to)
    #Group it by day and count the number
    data1_grouped=pd.DataFrame(data1.groupby('date').retweets.agg('sum')).reset_index()
    #Rename columns to reference them better
    data1_grouped.columns = ['Date', 'RetweetsNumber']

    # plot
    trace1 = go.Scatter(
        x=data1_grouped['Date'],
        y=data1_grouped['RetweetsNumber'],
        name=label1
    )

    data = [trace1]
    #Process data2
    if not data2.empty:
        #Group it by day and count the number
        data2=date_restrictions(data2, date_to)
        #Ignore dates before date_to
        data2_grouped=pd.DataFrame(data1.groupby('date').retweets.agg('sum')).reset_index()
        #Rename columns
        data2_grouped.columns = ['Date', 'RetweetsNumber']

        #Plot
        trace2 = go.Scatter(
            x=data2_grouped['Date'],
            y=data2_grouped['RetweetsNumber'],
            name=label2
        )
        data.append(trace2)
    #Process events
    if not events.empty:
        #Ignore dates before date_to
        events=date_restrictions(events, date_to)

        #Add markers
        trace3 = go.Scatter(
            x=list(events['date']),
            y=np.ones(len(events)) * 5,
            mode='markers',
            text=list(events['event']),
            name='events',
            marker = dict(
                size = 10,
                line = dict(
                    width = 2,
                )
            )
        )
        data.append(trace3)

    #adding the events
    layout = go.Layout(
        showlegend=True,
        title = 'Number of retweets of {} and {} over time'.format(label1,label2)
    )
    fig = go.Figure(data=data, layout= layout)
    plot_url = py.plot(fig, filename='simple-annotation')


# <a id='language'></a>
# ## Language analysis
#
# The functions defined in this section perform a basic analysis on the language found and translate the tweets that are not in english.
#
# For a first analysis of the language we can find in our datasets we define the function *lang_analysis* which returns the number and percentage of the tweets in each language. In addition, it also prints 3 randomly chosen tweets

# In[2]:

def lang_analysis(df):

    lang_count=[]
    #Repeat analysis for all languages
    for lang in np.unique(df.lang.dropna()):
        tweets_lang=df.loc[df['lang'] == lang]
        #Print the number of tweets
        print('\033[1m')
        print('Number of tweets in {}: {}, {}%'.format(lang, len(tweets_lang), len(tweets_lang)/len(df)))
        print('\033[0m')
        #Print 3 random tweets of each language
        print('\n'.join(np.random.choice(tweets_lang.text.values,3)))
        #Print horizontal line
        print('.'*130)
        lang_count.append(len(tweets_lang)/len(df))
    return lang_count



# Even though the percentage of non-english tweets is very low, we still believe it may be important for obtaining the topics. Thus, we implement a function *translate* which translates a tweet into the given language. This translatation uses the API from googletrans. Given the percentage of tweets incorrectly tagged with another language when they are really in english, we find in which language is the tweet written and only translate the tweets that are not in english. It should also be mentioned that hashtags are not translated.

# In[25]:


def translate(tweet, lang_id):
    '''
    Translate a given tweet to lang_id
    INPUT:
        tweet: text to translate
        lang_id: language id
    OUTPUT:
        tweet translated
    '''
    translator = Translator()

    #If tweet is in same language, do not translate
    if (translator.detect(tweet).lang==lang_id):
        return tweet

    #Return translation
    return translator.translate(tweet, dest=lang_id).text


# Finally, due to the missing language id in the tweets from the _GetOldTweets_ API we define the function *detect_language* which given a language returns its id.

# In[30]:

def detect_language(tweet):
    '''
    Detect the language of curent tweet
    INPUT:
        tweet: text to translate
    OUTPUT:
        lang_id: language id
    '''
    translator = Translator()

    #If tweet is in same language, do not translate
    return translator.detect(tweet).lang




# <a id='topics'></a>
# ## Topics modelling
#
# This section contains the functions used for obtaining the LDA model for topic modelling and for obtaining the topic of a tweet given the model
#
# With topic modelling, we automatically identify the topics present in the tweets. A topic is a bunch of words that tend to repeat together in a collection of tweets (corpus). We use this approach over using keywords, because we believe that as topics combine several words, they express ideas better than just one term.
#
# We use LDA (Latent Dirichlet Allocation) as our modelling tecnique to obtain lists of topics. LDA assumes that each document, or tweet, in our case is produced from a mixture of topics. Each topic is a mixture of topic terms with different weights. It has three parameters, the number of topics, the number of terms in each topic and the maximum number of iterations until convergence.
#
# For the topic extraction, we followed the pipeline described in https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/. We use the LDA implementation from Gensim for topic modelling. We define the function *get_topics_list*, which given the dataframe containing the tweets, the number of topics to find and the preprocessing options, obtains the LDA model. In the first place, this functions calls the function defined above for the preprocessing and tokenization of the tweet. With the resulting corpus, we create the dictionary of terms and we convert it to the Document Term Matrix using the function *doc2bow*. This matrix contains the frequency of each word in each document. With this matrix, we train the LDA model calling the function *Lda*.
#

# In[21]:

def get_topics_list(tweet_df, num_topics=50, handles=True, length=True, stop=True, punctuation=True,
                     numbers=True, urls=True, retweet=True, slang=True, path_slang='./'):
    '''
    Return num_topics topics found in the tweets from tweet_df using LDA
    INPUT:
        tweet_df: dataframe containing the tweets as returned by process_tweets
        num_topics: number of topics to find
        handles: preprocessing option, if true, remove handles (user mentions)
        length: preprocessing option, if true, reduce length of more than three characters repeated to three characters, e.g, cooool -> coool
        stop: preprocessing option, if true, remove stopwords
        punctuation: preprocessing option,  true, remove punctuation
        numbers: preprocessing option, if true, if true, remove numbers
        urls: preprocessing option, if true, remove urls.
        retweet: preprocessing option, if true, delete RT tag.
        slang: preprocessing option, if true, replace slang words
        path_slang: preprocessing option, path to the pickled slang dictionary
    OUTPUT:
        trained lda model. With show_topics, each line is a topic with individual topic terms and weights.
        dictionary used to generate the model
    '''

    #Function obtained following https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/

    #Preprocessing for each tweet in the dataframe
    doc_clean = [preprocess_tweet(tweet, handles, length, stop, punctuation,
                     numbers, urls, retweet, slang) for tweet in tweet_df.text.values]

    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


    params = {'passes': 10, 'random_state': seed} #10 passes from each word
    # Running and Training LDA model on the document term matrix.
    ldamodel = LdaMulticore(corpus=doc_term_matrix, num_topics=num_topics, id2word=dictionary, workers=6,
                    passes=params['passes'], random_state=params['random_state'])

    return ldamodel, dictionary, doc_term_matrix



# Next, we define a function that given a tweet, the LDA model and the dictionary used to generate it, obtains the underlying topics of the tweet and the probability of each of the topics. In the first place, we preprocess the tweet as before, and convert it to the Document Term Matrix using the dictionary. Finally, we obtain the list of topics for the tweet, with the format (idx, probability), where idx refers to the index of topic in the lda model.

# In[14]:

def get_tweet_topic(tweet, ldamodel, dictionary, handles=True, length=True, stop=True, punctuation=True,
                     numbers=True, urls=True, retweet=True, slang=True, path_slang='./'):

    '''
    Obtain the topic of an unseen tweet.
    INPUT:
        tweet_df: dataframe containing the tweets as returned by process_tweets
        ldamodel: model of lda as returned by get_topics_list
        dictionary: dictionary that created the lda model
        handles: preprocessing option, if true, remove handles (user mentions)
        length: preprocessing option, if true, reduce length of more than three characters repeated to three characters, e.g, cooool -> coool
        stop: preprocessing option, if true, remove stopwords
        punctuation: preprocessing option,  true, remove punctuation
        numbers: preprocessing option, if true, if true, remove numbers
        urls: preprocessing option, if true, remove urls
        retweet: preprocessing option, if true, delete RT tag.
        slang: preprocessing option, if true, replace slang words
        path_slang: preprocessing option, path to the pickled slang dictionary
    OUTPUT:
        topic obtained with format (idx, probability) where idx is the index of the topic in the ldamodel

    '''
    #Preprocess tweet
    doc_clean = preprocess_tweet(tweet, handles, length, stop, punctuation, numbers, urls, retweet, slang)

    #Convert to document term matrix
    doc_term = dictionary.doc2bow(doc_clean)

    #Obtain the topic. Each line represents a topic and the probability
    topic=ldamodel[doc_term]

    #Return topic
    return topic


# <a id='hashtag'></a>
# ## Hashtags analysis
#
# In this section we have two functions, one for analysis the coocurrence of hashtags over time and another one for generating wordclouds of hashtags. In addition to this functions, we also create an interactive graph to show the communities that can be found in the hashtag and how they are tipically linked. However, this graph is not defined in this notebook and can be found in ....
#
# _Polarization: Aparition of hashtags against and in favour together_
#
# We analyse the aparition of hashtags that are considered supportive (e.g., #BlackLivesMatter) and against (e.g., #AllLiveMatters, #PoliceLivesMatter, etc.) of the protest together, to understand whether this hashtags were not as polarized (in favour or against) as it is normally thought or whether this polarization grew over time.
#
# For this purpose, we create the function *coocurrence_hashtag* which obtains which hashtags against the protest are used with the hashtags in support. First, we obtain which of the hashtags against the protest where used with this hashtag by looking in the *hashes_inside* column for the other hashtags and adding them to a new column called *other_tag* in a new dataframe called *cooc_tags*. Then, we plot the percentage of tweets with both a hashtag for and against the protest in a bar plot, aggregated by month.

# In[15]:

def cooccurence_hashtag(data, antitags, label, plot=True):
    '''
    Obtain tweets with co-occurence of two hashtags.
    INPUT:
        data: dataframe where the co-occurence has to be analysed
        antitags: tags to look for in hashes_inside
        label: principal hashtag we are analysing for the title
        plot: if true, plot which percentage of tweets have both hashtags aggregated by month

    OUTPUT:
        co-occurence dataframe with the same columns as usual and an additional column 'other_tag' which indicates which
        tag in addition to the principal has appeared
    '''
    #Copy dataframe adding a column for the other tag found
    cooc_tags=data.copy()
    cooc_tags["other_tag"] = np.nan
    #For each tag, mark the tweets where we find the hashatg with it
    for tag in antitags:
        #Put tag (as lowercase) as other_tag for the tweets were tag in the other hashes, without being case sensitive.
        cooc_tags.loc[data['hashes_inside'].str.contains(tag,case=False), 'other_tag']=tag.lower()
    #Drop rows without any of the antitags in the text
    cooc_tags=cooc_tags.dropna()

    #Bar plot with percentages
    if plot:
        #Group the dates by month
        cooc_tags['date_month']=cooc_tags.date.dt.to_period("M")
        #Obtain the total number of tweets we have per month
        data_by_month=data.date.dt.to_period("M")
        #Only save information where we have data from both monts
        data_by_month=data_by_month[data_by_month.isin(cooc_tags.date_month)]

        #Aggregate data by month and tag
        tags_grouped=cooc_tags.groupby(['date_month', 'other_tag']).size().unstack()


        #Bar plot with percentages: divide by the total data by month we have
        ax=tags_grouped.div(data_by_month.groupby(data_by_month).size().values, axis=0).plot(kind='bar',
                                                                                       stacked=False,
                                                                                       figsize=[15,4],
                                                                                       grid=True)
        #Legend outside the plot
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        #Ticks and labels
        ax.set_title('Co-occurence of {} hashtags over time'.format(label),fontsize=14, fontweight='bold') #Title
        ax.set_xlabel('Month',fontsize=16, fontweight='bold') #xlabel
        ax.set_ylabel('Percentage',fontsize=16, fontweight='bold') #ylabel
        ax.tick_params(labelsize=12) #Size

    return cooc_tags


# Interactive version

# In[ ]:

def cooccurence_hashtag_interactive(data, antitags, label1, label2, plot=True):
    '''
    Obtain tweets with co-occurence of two hashtags (interactive)
    INPUT:
        data: dataframe where the co-occurence has to be analysed
        antitags: tags to look for in hashes_inside
        label1: principal hashtag we are analysing for the title
        label2: second hashtag we are analysing for the title
        plot: if true, plot which percentage of tweets have both hashtags aggregated by month

    OUTPUT:
        co-occurence dataframe with the same columns as usual and an additional column 'other_tag' which indicates which
        tag in addition to the principal has appeared
    '''
    #Copy dataframe adding a column for the other tag found
    cooc_tags=data.copy()

    cooc_tags["other_tag"] = np.nan
    #For each tag, mark the tweets where we find the hashatg with it
    for tag in antitags:
        #Put tag (as lowercase) as other_tag for the tweets were tag in the other hashes, without being case sensitive.
        cooc_tags.loc[data['hashes_inside'].str.contains(tag,case=False), 'other_tag']=tag.lower()

    #Drop rows without any of the antitags in the text
    cooc_tags=cooc_tags.dropna()


    #Bar plot with percentages
    if plot:
        #Group the dates by month
        cooc_tags['date_month']=cooc_tags.date.dt.to_period("M")
        #Obtain the total number of tweets we have per month
        data_by_month=data.date.dt.to_period("M")

        #Only save information where we have data from both months
        data_by_month=data_by_month[data_by_month.isin(cooc_tags.date_month)]


        #Aggregate data by month and tag
        tags_grouped=cooc_tags.groupby(['date_month', 'other_tag']).size().unstack()
        #Obtain percentage
        tags_grouped=pd.DataFrame(tags_grouped.div(data_by_month.groupby(data_by_month).size().values, axis=0)).reset_index()
        plot_data=[]

        #One bar per each of the plots
        for tag in tags_grouped.columns[1:]:

            trace1 = go.Bar(
                x=tags_grouped['date_month'].dt.to_timestamp(),
                y=tags_grouped[tag].fillna(0),
                name=tag
            )
            plot_data.append(trace1)

         #Plots
        layout = go.Layout(
            showlegend=True,
            yaxis = dict(title='Percentage'),
            title = 'Co-occurence of {} and {} hashtags over time'.format(label1,label2)
        )
        fig = go.Figure(data=plot_data, layout= layout)
        plot_url = py.plot(fig, filename='occurence'+label1)



    return cooc_tags


# <a id='sentiment'></a>
# ## Sentiment analysis
#
# We perform sentiment analysis each tweet using the TextBlob. With this study we can classify tweets as:
# * Positive: if polarity is bigger than 0
# * Negative: if polarity is smaller than 0
# * Neutral: if polarity is 0
#
# In addition, we also obtain the subjectivity of the tweet which indicates whether the tweet is objective or rational (subjectivity < 0.5); subjective or emotional (subjectivity > 0.5) or neutral (subjectivity=0.5). This allows an analysis of how people react and how emotionally envolved are they.
#
# We create the function *get_tweet_sentiment*, which given a tweet returns its polarity and subjectivity. It first preprocesses the tweet, without removing the retweet tag or puntuactions, as they may be informative of the sentiment. In addition, we do not remove stopwords as they are removed by TextBlob API and convert the list of tokens to a string separated by spaces. Once preprocessed, we call TextBlob to do the analysis of the clean tweet and return the polariy and subjectivity.

# In[16]:


def get_tweet_sentiment(tweet):
    '''
    Get the sentiment of a tweet
    INPUT:
        tweet: text
    OUTPUT:
        polarity (-1 to 1) of the tweet
        subjectivity(0 to 1) of the tweet
    '''
    #Preprocess tweet: do not remove punctuations or retweet tag. stopwords are removed by TextBlob
    preprocessing_options_sentiment={'punctuation':False, 'retweet':False, 'stop':False}
    clean_tweet=' '.join(preprocess_tweet(tweet, **preprocessing_options_sentiment)) #Convert list of tokens to string

    # Create TextBlob object passing it the preprocessed tweet
    analysis = TextBlob(clean_tweet)

    return analysis.sentiment.polarity, analysis.sentiment.subjectivity


# We plot both the mean polarity and subjectivity of both groups of the protest aggregated by week, in order to have more consistent trends than by day. We aggregate data depending on the metric passed as a parameter, so if data is more variable we can use the median, while if it's more stable we can use the mean. The rest of the definition follows the same signature as when plotting the number of tweets and retweets.

# In[17]:

def plot_sentiment_tweets(sentiment,data1, label1, data2=pd.DataFrame(), label2='', metric='mean',
                          events=pd.DataFrame(), date_to=''):
    '''
    Plot of the sentiment of tweets over time, grouped by week and agreggated with the given metric
    INPUT:
        sentiment: name of the column with the sentiment analysis data
        data1: principal data to plot.
        label1: label of data1
        data2: secondary data (tweets against protest)
        label2: label of data2
        metric: metric to aggregate data with
        events: dataframe with the events
        date_to: string with format YYYY-MM-DD, ignore data previous to that date
    '''
    #Ignore dates before date_to
    data1=date_restrictions(data1.copy(), date_to)

    #Process data 1
    data1['date_week']=data1.date.dt.to_period("W") #Aggregate by week

    #Group it by day and count the number
    data1_grouped=data1.groupby('date_week')[sentiment].agg(metric)

    #Plot data
    ax=data1_grouped.plot(kind='line',
                         figsize=[15,6],
                         label=label1,
                         legend=True,
                         marker='|')

    #Process data2
    if not data2.empty:
        data2=date_restrictions(data2.copy(), date_to)
        data2['date_week']=data2.date.dt.to_period("W") #Aggregate by week
        #Group it by day and count the number

        #Ignore dates before date_to
        data2_grouped=data2.groupby('date_week')[sentiment].agg(metric)
        #Plot data
        data2_grouped.plot(ax=ax,
                          kind='line',
                          label=label2,
                          legend=True,
                          style='r',
                          marker='|'
                          )
    #Process events
    if not events.empty:
        events_c=events.copy()
        #Ignore dates before date_to
        events_c=date_restrictions(events_c, date_to)
        events_c['date']=events_c.date.dt.to_period("W") #Aggregate by week


        #Plot events on top of existing plot
        plot_events(events_c,data1_grouped, offset=0.001)
        del events_c

    #Labels
    #Title with full name of the column
    if sentiment=='subj':
        title_str='Subjectivity'
    else:
         title_str='Polarity'
    ax.set_title('{} of tweets of {} protest over time'.format(title_str, label1),fontsize=14, fontweight='bold') #Title
    ax.set_xlabel('Date',fontsize=16, fontweight='bold') #xlabel
    ax.set_ylabel(title_str,fontsize=16, fontweight='bold') #ylabel

    #Change ticks parameters
    ax.tick_params(labelsize=10) #Size
    ax.xaxis.set_major_formatter(dates.DateFormatter('%d\n%b\n%Y')) #Format




# Interactive version

# In[ ]:

def plot_sentiment_tweets_interactive(sentiment, data1, label1, data2=pd.DataFrame(), label2='', events=pd.DataFrame(), date_to='', metric='mean'):
    '''
    Plot of the sentiment of tweets over time, grouped by week and agreggated with the given metric
    INPUT:
        sentiment: name of the column with the sentiment analysis data
        data1: principal data to plot.
        label1: label of data1
        data2: secondary data (tweets against protest)
        label2: label of data2

        events: dataframe with the events
        date_to: string with format YYYY-MM-DD, ignore data previous to that date
        metric: metric to aggregate data with
    '''

    if sentiment=='subj':
        title_str='Subjectivity'
    else:
        title_str='Polarity'

    #Process data 1
    #Ignore dates before date_to
    data1=date_restrictions(data1, date_to)

     #Group the dates by week
    data1['date_week']=data1.date.dt.to_period("W") #Aggregate by week

    #Aggregate data by week with metric
    data1_grouped=pd.DataFrame(data1.groupby('date_week')[sentiment].agg(metric)).reset_index()
    #Rename columns
    data1_grouped.columns = ['Date', 'Sentiment']

    #Trace for principal data
    trace1 = go.Scatter(
        x=data1_grouped['Date'].dt.to_timestamp(),
        y=data1_grouped['Sentiment'],
        name=label1
    )

    #Add to data list
    data=[trace1]

    #Process data2
    if not data2.empty:
        data2=date_restrictions(data2, date_to)
        #Group it by day and count the number
        data2['date_week']=data2.date.dt.to_period("W") #Aggregate by week

        #Aggregate data by week with metric
        data2_grouped=pd.DataFrame(data2.groupby('date_week')[sentiment].agg(metric)).reset_index()
        #Rename columns
        data2_grouped.columns = ['Date', 'Sentiment']
        #Plot
        trace2 = go.Scatter(
            x=data2_grouped['Date'].dt.to_timestamp(),
            y=data2_grouped['Sentiment'],
            name=label2
        )

        #Add to data list
        data.append(trace2)

    #Process events
    if not events.empty:
        events_c=date_restrictions(events.copy(), date_to)
        events_c['date']=events_c.date.dt.to_period("W").dt.to_timestamp() #Aggregate by week

        trace3 = go.Scatter(
            x=list(events_c['date']),
            y=np.zeros(len(events_c)),
            mode='markers',
            text=list(events_c['event']),
            name='events',
            marker = dict(
                size = 10,
                line = dict(
                    width = 2,
                )
            )
        )
        #adding the events
        data.append(trace3)

    #Plots
    layout = go.Layout(
        showlegend=True,
        title = '{} of tweets of {} and {} over time'.format(title_str, label1, label2),
        yaxis = dict(title=title_str)
    )
    fig = go.Figure(data=data, layout= layout)
    plot_url = py.plot(fig, filename=title_str+label1)


# With the function *obtain_tweet_sentiment* we obtain the type of each tweet according to the sentiment passed as parameter, that is, rational, neutral or subjective if we are analysing subjectivity and postive, negative or neutral if we are analysing polarity. In addition, we also print the statistics (percentage of each type, mean and std when they are not neutral).

# In[36]:

def obtain_type_sentiment(data, sentiment, label):
    '''
    Obtain the type of the sentiment of each tweet
    INPUT:
        sentiment: name of the column with the sentiment analysis data
        data: dataframe
        label: label of protest
    OUTPUT:
        dataframe with added column type_sent with type of sentiment: 1 (positive or rational), 0 (neutral), -1 (negative or
        subjective)
    '''
    #Specify labels and limit to distinguish the types
    if sentiment=='subj':
        title_str='Subjectivity'
        limit=0.5
        neg_str='rational'
        pos_str='subjective'
    else:
        limit=0
        title_str='Polarity'
        pos_str='positive'
        neg_str='negative'

    data=data.copy()
    #Initalise column
    data['type_sent']=np.nan
    #Positive
    data.loc[data[sentiment]>limit, 'type_sent']=1
    #Neutral
    data.loc[data[sentiment]==limit, 'type_sent']=0
    #Negative
    data.loc[data[sentiment]<limit, 'type_sent']=-1

    #Number of total tweets
    n_data=len(data)


    #Statistics
    print('\033[1m'+'{} of {}'.format(title_str, label)+'\033[0m')
    print("Percentage of {} tweets: {}%".format(pos_str, 100*len(data[data['type_sent']==1])/n_data))
    print("    Mean: {}\n    Std: {} ".format(np.mean(data.loc[data['type_sent']==1, sentiment]),
                                               np.std(data.loc[data['type_sent']==1,sentiment])))
    print("Percentage of neutral tweets: {}%".format(100*len(data[data['type_sent']==0])/n_data))
    print("Percentage of {} tweets: {}%".format(neg_str, 100*len(data[data['type_sent']==-1])/n_data))
    print("    Mean: {}\n    Std: {} ".format(np.mean(data.loc[data['type_sent']==-1, sentiment]),
                                               np.std(data.loc[data['type_sent']==-1, sentiment])))

    return data


# *plot_type_sentiment* plots the percentage of tweets of each type of the sentiment passed as parameter over time. If events are passed, they are added to the plot. In addition, if a date is given, previous dates data is ignored and not plotted.
#
# In this case, we only plot one dataframe per plot, as each plot already has 3 lines and adding more would make it more difficult to read.

# In[37]:

def plot_type_sentiment(sentiment,data, label, events=pd.DataFrame(), date_to=''):
    '''
    Obtain the percentage of tweets of each type of sentiment
    INPUT:
        sentiment: name of the column with the sentiment analysis data
        data: dataframe
        label: label of protest
        events: dataframe with the events
        date_to: string with format YYYY-MM-DD, ignore data previous to that date
    OUTPUT:
        dataframe with added column type_sent with type of sentiment: 1 (positive or rational), 0 (neutral), -1 (negative or
        subjective)
    '''
    #Ignore dates before date_to
    data=date_restrictions(data.copy(), date_to)

    #Categorize sentiment in positive, neutral and negative
    data=obtain_type_sentiment(data, sentiment, label)


    #Group the dates by week
    data['date_week']=data.date.dt.to_period("W") #Aggregate by week

    #Aggregate data by month and tag, and divide by the total data of month
    sent_grouped=data.groupby(['date_week', 'type_sent']).size().unstack().div(
                                                                            data.groupby('date_week').size().values,
                                                                            axis=0)

    if sentiment=='subj':
        leg=['Subjective','Neutral','Rational']
        title_str='Subjectivity'
    else:
        title_str='Polarity'
        leg=['Positive','Neutral', 'Negative']
    #Bar plot with percentages
    ax=sent_grouped.plot(kind='line', figsize=[15,4], grid=False, marker='|')
    #Legend outside the plot
    plt.legend(leg,loc='center left', bbox_to_anchor=(1.0, 0.5))

    if not events.empty:

        events_c=events.copy()
        #Ignore dates before date_to
        events_c=date_restrictions(events_c, date_to)
        events_c['date']=events_c.date.dt.to_period("W") #Aggregate by week

        print('\033[1m'+'Events '+'\033[0m')
        #Plot vertical line in each event date
        for i, xc in enumerate(events_c['date']):
            plt.axvline(x=xc, linestyle='--', color='k', label=i, linewidth=0.5)
            print(i, events_c.iloc[i]['event'])
            plt.text(xc, 0.2, i)


        #Ticks and labels
        ax.set_title('Distribution of {} of {} tweets over time'.format(title_str, label),fontsize=14, fontweight='bold') #Title
        ax.set_xlabel('Month',fontsize=14, fontweight='bold') #xlabel
        ax.set_ylabel('Percentage',fontsize=14, fontweight='bold') #ylabel
        ax.tick_params(labelsize=12) #Size

    return data


# *plot_type_sentiment_interactive* is an alternative to the previous function, which uses and interactive plot form plotly instead of being static and inline.

# In[20]:

# alternative function to make it interactive
def plot_type_sentiment_interactive(sentiment,data, label, events=pd.DataFrame(), date_to=''):
    '''
    Interactive plot of the percentage of tweets of each type of sentiment
    INPUT:
        sentiment: name of the column with the sentiment analysis data
        data: dataframe
        label: label of protest
        events: dataframe with the events
        date_to: string with format YYYY-MM-DD, ignore data previous to that date

    '''
    #Ignore dates before date_to
    data=date_restrictions(data.copy(), date_to)

    #Categorize sentiment in positive, neutral and negative
    data=obtain_type_sentiment(data, sentiment, label)

    #Group the dates by week
    data['date_week']=data.date.dt.to_period("W") #Aggregate by week

    #Aggregate data by month and tag, and divide by the total data of month
    sent_grouped=data.groupby(['date_week', 'type_sent']).size().unstack().div(
                                                                            data.groupby('date_week').size().values,
                                                                            axis=0).reset_index()




    if sentiment=='subj':
        sent_grouped.columns = ['Date','Subjective','Neutral','Rational']
        title_str='Subjectivity'
    else:
        title_str='Polarity'
        sent_grouped.columns = ['Date','Positive','Neutral', 'Negative']

    #Process events
    if not events.empty:
        #Ignore dates before date_to
        events_c=date_restrictions(events.copy(), date_to)
        events_c['date']=events_c.date.dt.to_period("W").dt.to_timestamp() #Aggregate by week

    # plot
    #Trace of rational/postive
    trace1 = go.Scatter(
        x=sent_grouped['Date'].dt.to_timestamp(),
        y=sent_grouped[sent_grouped.columns[1]], #First of three categories +date column
        name=sent_grouped.columns[1]
    )
    #Trace of neutral
    trace2 = go.Scatter(
        x=sent_grouped['Date'].dt.to_timestamp(),
        y=sent_grouped[sent_grouped.columns[2]],
        name=sent_grouped.columns[2]
    )
    #Trace of emotional/negative
    trace3 = go.Scatter(
        x=sent_grouped['Date'].dt.to_timestamp(),
        y=sent_grouped[sent_grouped.columns[3]],
        name=sent_grouped.columns[3]
    )

    #Add to data list
    data = [trace1, trace2, trace3]
    #Add data of events
    if not events.empty:
        #Add a vertical line per event
        shapes=[]
        for row in events_c['date']:

            shapes.append({'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': row,
                           'y0': 0,
                           'x1': row,
                           'y1': 1,
                           'opacity': 0.4})

        #Trace
        trace_events = go.Scatter(
            x=list(events_c['date']),
            y=np.zeros(len(events)),
            mode='markers',
            text=list(events_c['event']),
            name='events',
            marker = dict(
                size = 10,
                line = dict(
                    width = 2,
                )
            )
        )

        #Add events
        data.append(trace_events)

    #adding the events
    layout = go.Layout(
        barmode='group',
        showlegend=True,
        title = 'Distribution of {} of {} tweets over time'.format(title_str, label),
        yaxis=dict(title='Percentage'),
        shapes=shapes
    )
    fig = go.Figure(data=data, layout= layout)
    plot_url = py.plot(fig, filename=title_str+label)

def smooth(x,window_len=5,window='blackman'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

'''
similar function to deal with different time scales
'''
def plot_num_tweets2(data1, label1, data2=pd.DataFrame(), label2='', events=pd.DataFrame(), date_to='', weekday=1, interval=1):
    '''
    Plot of the number of tweets over time, grouped by day
    INPUT:
        data1: principal data to plot.
        label1: label of data1
        data2: secondary data (tweets against protest)
        label2: label of data2
        events: dataframe with the events
        date_to: string with format YYYY-MM-DD, ignore data previous to that date
    '''
    #Process data 1
    #Ignore dates before date_to
    data1=date_restrictions(data1, date_to)
    #Group it by day and count the number
    data1_grouped=data1.groupby('date').size()

    #Plot data
    ax=data1_grouped.plot(kind='line',
                         figsize=[15,6],
                         label=label1,
                         legend=True)

    #Process data2
    if not data2.empty:
        #Group it by day and count the number
        data2=date_restrictions(data2, date_to)
        #Ignore dates before date_to
        data2_grouped=data2.groupby('date').size()
        #Plot data
        data2_grouped.plot(ax=ax,
                          kind='line',
                          label=label2,
                          legend=True,
                          style='r',
                          )
    #Process events
    if not events.empty:
        #Ignore dates before date_to
        events=date_restrictions(events, date_to)
        #Plot events on top of existing plot
        plot_events(events,data1_grouped)

    #Labels
    ax.set_title('Number of tweets of {} protest over time'.format(label1),fontsize=14, fontweight='bold') #Title
    ax.set_xlabel('Month',fontsize=16, fontweight='bold') #xlabel
    ax.set_ylabel('Frequency',fontsize=16, fontweight='bold') #ylabel
    #Change ticks parameters
    ax.tick_params(labelsize=8) #Size
    ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(weekday), interval=interval)) #Position
    ax.xaxis.set_major_formatter(dates.DateFormatter('%d\n%b\n%Y')) #Format
