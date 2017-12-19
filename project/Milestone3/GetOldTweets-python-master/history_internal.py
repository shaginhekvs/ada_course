# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

from sqlalchemy import *


class Deal_tweets():
    
    def __init__(self,path):
        self.engine=create_engine('sqlite:///%s'%path)
    
    

    # Example 2 - Get tweets by query search
    def read_tweets(self,table_name,string_search,start_time,end_time,num_per_6H=1000,test_if_start_time_read=False):
        engine=self.engine
        try:
            df2=pd.read_sql_query('SELECT date FROM %s'%(table_name),engine,parse_dates='date')
            if(len(df2)>0):
                start_time=df2.date.max().to_pydatetime()
                pass
        except:
            pass
        
        self.create_sqllite_table(table_name,engine)
        print 'reading from %s to %s'%(start_time,end_time) 

        start_time=pd.Timestamp(start_time,tz='Europe/Zurich')
        end_time=pd.Timestamp(end_time,tz='Europe/Zurich')
        time_list=list(pd.date_range(start_time,end_time,freq='6H').tz_convert('Europe/Zurich'))
    
        if(len(time_list)<=1):
            print 'time too short, enter time like yyyy-dd-mm-hh-mm-ss'
            return
        
        for i_str,i_end in zip(time_list[:-1],time_list[1:]):
            print 'reading from %s to %s'%(i_str,i_end)
            df=pd.DataFrame()
            since = i_str.to_pydatetime().strftime('%Y-%m-%d-%H-%M-%S')
            end = i_end.to_pydatetime().strftime('%Y-%m-%d-%H-%M-%S')
            
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch(string_search).setSince(since).setUntil(end).setMaxTweets(num_per_6H)
            tweets = got.manager.TweetManager.getTweets(tweetCriteria)
            if(len(tweets)==0):
                continue
            else:
                for tweet in tweets:
                    df2=pd.read_sql_query('SELECT * FROM %s WHERE  %s.id==%s'%(table_name,table_name,tweet.__dict__['id']),engine,parse_dates='date',index_col='date')
                    if(len(df2)>0):
                        continue
                    df=df.append(pd.DataFrame(dict(map(lambda (x,y) : (x,[y]) , tweet.__dict__.iteritems()))))
                cols_to_keep=['date','hashtags','geo','id','mentions','retweets','text','username']
                cols_to_rem=list(set(df.columns)-set(cols_to_keep))
                df.drop(cols_to_rem,axis=1,inplace=True)
            
            df.to_sql(table_name,con=engine,if_exists='append',index=False)
    
                
            
        print('read from %s to %s'%(start_time,end_time))
        
    def create_sqllite_table(self,name,db):
     
         #currently just use for sync_selected
         metadata=MetaData()
         cols=()
         
         col_str_date=Column('date',sqlalchemy.DATETIME(timezone=True))
         
         col_hash=Column('hashtags',sqlalchemy.String(300))
         
         col_geo=Column('geo',sqlalchemy.String(300))
         
         col_id=Column('id',sqlalchemy.String(40),primary_key=True)
         
         col_mention=Column('mentions',sqlalchemy.String(300))
         
         col_re=Column('retweets',sqlalchemy.String(20))
         
         col_text=Column('text',sqlalchemy.String(300))
         
         col_user=Column('username',sqlalchemy.String(40))
         
         cols=(col_str_date,col_hash,col_geo,col_id,col_mention,col_re,col_text,col_user)
         
         table=Table(name,metadata,col_str_date,col_hash,col_geo,col_id,col_mention,col_re,col_text,col_user)   
         
         table.create(db,checkfirst=True)
         


    
                
            
    