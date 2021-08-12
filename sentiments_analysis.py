# Import the libraries
import pandas as pd
import numpy as np
import datetime
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(1500)

#newapi library
from newsapi import NewsApiClient
#youtube api libraries
from youtube_easy_api.easy_wrapper import *

# Data Preprocessing and Feature Engineering
from textblob import TextBlob
import json
import csv
import num2word
import textblob
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# reddit libraries
import praw


class SentimentsAnalysis:

    # language detection
    def detect_language(self, text):
        blob = TextBlob(text)
        if (blob.detect_language() == 'en'):
            return True
        else:
            return False

    # Text cleaning
    def dataCleaning(self, text):
         from nltk.corpus import stopwords
         punctuation = string.punctuation
         stopwords = stopwords.words('english')
         text = text.lower()
         text = "".join(x for x in text if x not in punctuation)
         words = text.split()
         words = [w for w in words if w not in stopwords]
         text = " ".join(words)

         return text

    # Data Scrapping
    def get_news_data(self, keyword):
        # Initiating the news api client with key
        newsapi = NewsApiClient(api_key='8f2a9534ae26466f8ae4d0f12643f94c')

        # Using the client library to get articles related to search
        all_articles = newsapi.get_everything(q=keyword,
                                              language='en',
                                              sources='bbc-news,The New York Times',
                                              from_param='2021-07-11',
                                              to='2021-08-09',
                                              sort_by='relevancy')
        df = pd.DataFrame.from_dict(all_articles)
        # save updated data

        df = df.assign(Group=1)
        df1 = df.articles.apply(pd.Series)
        df2 = df1.source.apply(pd.Series)
        df3 = pd.concat([df, df1, df2], axis=1).drop(
          ['urlToImage', 'name', 'id', 'articles', 'source', 'status', 'totalResults'], 1)
        df3 = df3[['Group', 'publishedAt', 'author', 'url', 'title', 'description', 'content']]
        df3.to_csv('scrapped_news_data.csv', mode='a', header=True, index=False)

        newsdata = pd.read_csv("scrapped_news_data.csv")
        # print(newsdata.columns.tolist())

        newsdata["combined"] = (newsdata["title"] + newsdata["description"] + newsdata["content"])
        newsdata = newsdata.dropna(axis=0, subset=['combined'])
        newsdata['combined'].astype('str')
        newsdata['cleaned'] = newsdata['combined'].apply(self.dataCleaning)

        newsdata['polarity'] = newsdata['cleaned'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        newsdata['subjectivity'] = newsdata['cleaned'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
        newsdata.to_csv('news_sentiments.csv')

        cols_to_keep = ['title', 'url', 'content', 'description', 'polarity', 'subjectivity', 'publishedAt']
        df = newsdata[cols_to_keep]
        news_json = df.to_json(orient='records', default_handler=str)
        return news_json

    #Reddit data scrapping
    def get_reddit_data(self, keyword):
        #reddit api authorization
        reddit = praw.Reddit(client_id='9HwE8uvXZeJhVc8yQg7ISQ', client_secret='LdgRC8E7u8CfnRAU7Ymb06l73aolmg',
                             user_agent='Muhammad Waheed Waqar')
        posts = []


        # scrapped data from reddit using reddit api
        for post in reddit.subreddit("all").search(keyword):
            posts.append([post.id, post.url, post.subreddit, post.title, post.selftext,
                         datetime.datetime.utcfromtimestamp(post.created), post.num_comments,
                         post.score, post.upvote_ratio, post.ups, post.downs])
            # Make dataframe
        posts = pd.DataFrame(posts, columns=['id', 'url', 'subreddit', 'title', 'body', 'date',
                                             'num_comments', 'score', 'upvote_ratio', 'ups', 'downs'])
       # Save orignal scrapped data in csv file
        posts.to_csv('scrapped_reddit_data.csv')

        # Clean title and make another coulmn to store cleaned title
        posts['cleaned_title'] = posts['title'].apply(self.dataCleaning)

        # calculate polarity and subjectivity of title using textblob
        posts['polarity'] = posts['cleaned_title'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        posts['subjectivity'] = posts['cleaned_title'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
        posts.to_csv('reddit_sentiments.csv')

        # columns which return as json object
        cols_to_keep = ['id', 'url', 'subreddit', 'title', 'body', 'date',
                        'num_comments', 'score', 'upvote_ratio', 'ups', 'downs', 'polarity', 'subjectivity']
        df = posts[cols_to_keep]
        reddit_json = df.to_json(orient='records', default_handler=str)
        return reddit_json

    def get_youtube_data(self, keyword):
        easy_wrapper = YoutubeEasyWrapper()
        easy_wrapper.initialize(api_key='AIzaSyB2_uJKGde2I41jn_g2ziOwFhX08nzGe7c')
        #data retrieved from youtube

        results = easy_wrapper.search_videos(search_keyword=keyword,
                                             order='relevance')

        result_data = pd.DataFrame(results)

        metadata = easy_wrapper.get_metadata(video_id=results[0]['video_id'])
        final_data = pd.json_normalize(metadata)
        for e in range(1, len(results)):
            metadata = easy_wrapper.get_metadata(video_id=results[e]['video_id'])
            data = pd.json_normalize(metadata)
            final_data = final_data.append(data)
        final_data['video_id'] = result_data['video_id']
        final_data['channel'] = result_data['channel']

        final_data['cleaned'] = final_data['title'].apply(self.dataCleaning)

        final_data['polarity'] = final_data['cleaned'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        final_data['subjectivity'] = final_data['cleaned'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
        final_data.to_csv('youtube_sentiments.csv')

        cols_to_keep = ['video_id', 'channel', 'title', 'publishedAt', 'statistics.viewCount', 'statistics.likeCount',
                        'statistics.dislikeCount', 'statistics.commentCount', 'polarity', 'subjectivity']

        df = final_data[cols_to_keep]
        youtube_json = df.to_json(orient='records', default_handler=str)

        return youtube_json





keyword = input("Enter keyword: ")
sentiments_analysis = SentimentsAnalysis()

print(" news Data")
print(sentiments_analysis.get_news_data(keyword))
print(" Reddit Data")
print(sentiments_analysis.get_reddit_data(keyword))
print(" youtube Data")
print(sentiments_analysis.get_youtube_data(keyword))