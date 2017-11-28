# Title
What’s the oxygen for mass movements? 

# Abstract
We are going to analyze the coverage of social movements in Twitter throughout the United States. #BlackLivesMatter changed the way we protest: hashtag activism was born. Twitter has created a new place to participate in social movements, as other events like the Arab Spring has shown. The impact and importance of any statement is now measured in how many people retweet it. News appear on twitter before they do in traditional medias like newspapers.  We plan to have insights in how users see, share and discuss these protests and movements and analyse different trends in the data.

# Research questions

**Impact**

* How many people are engaged in the protest? 

We study this observing how many different people tweeted.

* What is the impact of the protest? 

We study the number of tweets and retweets (obtained with the API) for hashtags both for and against the protests over time to measure the impact they have. We also plan on relating concrete events to rises in the trends and compare whether the rise is both in tweets from supporters or not depending on the event.

* Which events heighten the impact of a protest?

By relating a rise to a protest we can observe how much did such event affect the protest. For example, we have observed that for Black Lives Matter, LeBron James and other basketball players wearing a t-shirt supporting the protest or being the front cover of a magazine had a bigger impact than the death of a new person.

* What is the evolution of a protest in time ?  

We look at tweet polarity, number of tweets and number of retweets to assess how does the impact of a protest evolves with time. 
Hashtags

* Do both sides of the conflict (for and against) use the same hashtags?

We study which hashtags are used by people supporting the protest and against it and analyse which ones were used by both sides of the hashtags and which not, since people against a protest usually avoid using its hashtag as it constitutes a measure of its impact.
We obtain which hashtags from the non-supporters are used in tweets from the supporters and how did this trend change over time, maybe some hashtag started being used by all but some group started using it and it polarized towards one side of the protest.

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

We queried the API for location of the tweet (where it was generated) and location of the user. A large percentage of the tweet locations are missing (more than 30%) so we are pondering whether it remains relevant to use the location since the analysis will be bias. We will make further explorations to see if you can make conclusive results or the bias is too high for the next milestone. 


# Dataset

We use the Twitter dataset. However, we limit our focus to concrete protests. To make sure we could handle the data size, for this milestone we focus on the protest Black Lives Matter and filter the tweets which use the hashtags normally use for supporting it (#BlackLivesMatter) or going against it (#AllLivesMatter, #BlueLivesMatter, #AllLivesMatter, #WhiteLivesMattter, #NYPDLivesMatter and #PoliceLivesMatter). Given that it is probably one of the larger protests with more tweets, we feel like we can handle data this size, as we could just use pandas dataframes in our local computers instead of the cluster.
One of the problems we encountered is that tweets were in six different languages. However, an analysis showed that most of the tweets were in english. We found that a lot of the tweets are tagged incorrectly with another language when in reality they were in english or just composed of hashtags. Using the googletrans API we create a function that translates a tweet to english only if it detects that the tweets is in another language, to prevent translating tweets in english.
In addition to this database, we also use Tweepy API to query for additional information on the tweets. Given the id of each tweet that can be found in the database on the cluster, we obtain the number of retweets, the number of followers and the location (if available). However, since Tweepy has a limit of 900 queries per application every 15 minutes, we distribute this load through three different applications, one per member of the group, which allows us to query the API thrice as fast, simultaneously. With this approach, we can obtain results from the API quickly. 
Finally, we also use information available on the internet about the protests and the hashtags used, with websites like http://hashtagify.me/.


# A list of internal milestones up until project milestone 3

1. Complete the graph for the hashtags analysis making it interactive
2. Complete the Black Lives Matter analysis
3. Look at a timeline of events in the Black Lives Matter protest and identify the influence they had in: the impact, the topics and the sentiments
4. Complete the topics analysis with the aforementioned graph and the use of the already defined functions by obtaining the most important topics (for each side of the protest) and their popularity over time
5. Repeat all the analysis with other possible protests: We aim to analyse the data and compare the trends found with Black Lives Matter to other protests. If possible, as Black Lives Matter is considered the born of hashtag activism we want to compare a protest before this movement and after. To decide on the protest we have to analyse the data and observe if we have enough representative data, but among our options are:
-Against misogyny: #ILookLikeAnEngineer, #MeToo, #YesAllWomen
-#OccupyWallStreet
-#MuslimsAreNotTerrorist
-Related to Trump: #MAGA, #TakeAKnee, #TheResistance
For all of the protests the steps we’d have to make are:
-Look for information on the most used hashtags in favour and against the movement 
-Obtain our initial data with such hashtags
-Analyse the contents of the tweets and the hashtags to observe whether there are other important hashtags. If they are, also obtain the tweets for such hashtags
-Repeat the analysis done for Black Lives Matter
-Compare the results and trends to those of Black Lives Matter 





# Questions for TA

Can we use the Tweepy API instead of the Twitter dataset on the cluster? Given our approach of dividing the workload in three machines, the querying is fast, and we could obtain all information at once, instead of having to use the tweets ids obtained from the database to query the API to obtain information like the number of retweets.

 



