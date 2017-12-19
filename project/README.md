# Title
Anatomy of Hashtags: Deconstructing Social Media Activism.

# Abstract
In this work we are analyzing the coverage of social movements in Twitter throughout the United States. Twitter has created a new place to participate in social movements, as other events like the Arab Spring has shown. The impact and importance of any statement is now measured in how many people retweet it. News appear on twitter before they do in traditional medias like newspapers.  We plan to have insights in how users see, share and discuss these protests and movements and analyse different trends in the data. For that we concentrate on three different movements, related to racism (#BlackLivesMatter and related), misogyny (#metoo and related) and Trump campaign (#maga and #antimaga).

Because we have three very different topics, we centralised the backbone of our functions in the file Helpers.py and created a notebook for each of the three movements to show our results.
# Research questions

**Impact**

* How many people are engaged in the protest?

We study this by observing how many unique people tweeted. Observing this allows us to see how centralised the movement is (are there a few key people sustaining the movement or there is a large population).

* What is the impact of the protest?

We study the number of tweets and retweets (obtained with the API) for hashtags both for and against the protests over time to measure the impact they have. We also related concrete events to rises in the trends and compare whether the rise is both in tweets from supporters or not depending on the event.

* Which events heighten the impact of a protest?

By relating a rise to a protest we can observe how much did such event affect the protest. For example, we have observed that for Black Lives Matter, LeBron James and other basketball players wearing a t-shirt supporting the protest or being the front cover of a magazine had a bigger impact than the death of a new person.

* What is the evolution of a protest in time ?  

We look at tweet polarity, number of tweets and number of retweets to assess how does the impact of a protest evolves with time.

* Do both sides of the conflict (for and against) use the same hashtags?

It is to be noted that we have two types of movements: polarised, such as BLM and Trump and non-polarised, as is the case for misogyny, for there is not any strong movement against misogyny.  
For the polarised movements, we study which hashtags are used by people supporting the protest and against it and analyse which ones were used by both sides of the hashtags and which not, since people against a protest usually avoid using its hashtag as it constitutes a measure of its impact.
We obtain which hashtags from the non-supporters are used in tweets from the supporters and how did this trend change over time, maybe some hashtag started being used by all but some group started using it and it polarized towards one side of the protest.

For misogyny, we obtain valuable insights by observing how hashtags from long spanning movements (i.e. #everydaySexism) are used in correlation with trending movements (i.e. #meToo) in order to maintain participation.


* Which are more popular?
Given the limits of words in tweets, hashtags are the keywords of tweets, so we make a wordcloud showing the most popular keywords, which in general tend to coincide with the hashtags found in each tweet.

* Which hashtags are used together?
We make a graph that gives bigger weights on the edges the more they appear together. This way we can find clusters of hashtags used together and explore the relations between them, which in turn will help us find topics of discussion more easily.

**Topics**

We use two different techniques for exploring which topics are used in the tweets. First, we use topics modelling using LDA, which obtains topics as combination of words. In addition, we maken the aforementioned graph which allows us to consider topics in terms of the hashtags used. For the next milestone, we plan to make it interactive to observe the relations better

* Which topics are being discussed and how do they change over time?

We have defined a function to compute the topics being discussed over all tweets and then one to obtain what is the estimated topic is. We are going to compute around 10 topics, and plot their popularity as the number of tweets included in that topic, over time and try and see which ones are more popular and if this is a trend that changes over time.

* How do certain events change the topics under discussion?

Following the study of a concrete event on the impact, we want to observe if this trigger new topics to be taken into account.

Even though we have written most of the helper functions for this subject, we have not analysed the results yet, because we believe we need a more in depth analysis of the impact and events of a protest before, so we can relate to them.

**Sentiment analysis**
We use the TextBlob API to perform sentiment analysis

* Which is the predominant sentiment in the tweets from the supporters and detractors?
We analyse whether tweets are polarised towards being positive and negative and how do these emotions change over time with a plot. However we find that this emotions are not complex enough because most of the tweets are marked as neutral.

* Do subjective or rational arguments predominate in each side of the protest?

We study the subjectivity of the tweets to consider them either rational or emotional for each side of the protest. As these kinds of movements normally evoque very raw emotions, we feel that most of the tweets are emotional and a tweet with low subjectivity may indicate a non-supporter using the hashtag of the supporters.

* How do polarity of subjectivity change over time?

Again we aim to relate the changes to concrete events, aggregating them over time and using the mean or median (if the std is high)

**Location**

We queried the API for location of the tweet (where it was generated) and location of the user. A large percentage of the tweet locations are missing (more than 30%) so we are pondering whether it remains relevant to use the location since the analysis will be bias.
Update: we decided to drop the location for the percentage was too small in order to have any conclusive results.

# Dataset

We use the Twitter dataset. However, we limit our focus to concrete protests. To make sure we could handle the data size, for this milestone we focus on the protest Black Lives Matter and filter the tweets which use the hashtags normally use for supporting it (#BlackLivesMatter) or going against it (#AllLivesMatter, #BlueLivesMatter, #AllLivesMatter, #WhiteLivesMattter, #NYPDLivesMatter and #PoliceLivesMatter). Given that it is probably one of the larger protests with more tweets, we feel like we can handle data this size, as we could just use pandas dataframes in our local computers instead of the cluster.
One of the problems we encountered is that tweets were in six different languages. However, an analysis showed that most of the tweets were in english. We found that a lot of the tweets are tagged incorrectly with another language when in reality they were in english or just composed of hashtags. Using the googletrans API we create a function that translates a tweet to english only if it detects that the tweets is in another language, to prevent translating tweets in english.
In addition to this database, we also use Tweepy API to query for additional information on the tweets. Given the id of each tweet that can be found in the database on the cluster, we obtain the number of retweets, the number of followers and the location (if available). However, since Tweepy has a limit of 900 queries per application every 15 minutes, we distribute this load through three different applications, one per member of the group, which allows us to query the API thrice as fast, simultaneously. With this approach, we can obtain results from the API quickly.
Finally, we also use information available on the internet about the protests and the hashtags used, with websites like http://hashtagify.me/.
For Milestone 3, we also used an API for obtaining old tweets, because the Tweepy API only allowed us to get tweets from one week back, while the dataset in the cluster was not recent enough for the movements we wanted to study (on misogyny and Trump).
It should be noted however that this API gave a 1000 tweets upper bound on the number of tweets from a day. Nevertheless, we think that is a representative sample to observe trends over long-spanning periods. (and also makes it manageable)


# Contributions

Ada Pozo Perez conducted the exploratory analysis, designed the methodology for analysing the movements and studied in the depth the #BlackLivesMatter movement. Isabela Constantin designed the workflow to use the Tweepy API and dealt with presenting the information interactively, and studied in depth the misogyny related hashtags. Keshav Singh crawled most of the data, came up with the word graphs and studied in depth the maga-antinmaga movement.

For the final presentation, we would like to ask if it is possible that we each talk about the movement we focused on.
