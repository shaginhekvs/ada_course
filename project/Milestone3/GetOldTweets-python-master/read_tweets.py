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




# Example 2 - Get tweets by query search
def read_tweets(table_name,string_search,path,start_time,end_time,num_per_6H=1000,test_if_start_time_read=False):
    engine=create_engine('sqlite:///%s'%(path))
    try:
        
        df2=pd.read_sql_query('SELECT date FROM %s WHERE %s.text LIKE "%%%s%%"'%(table_name,table_name,string_search),engine,parse_dates='date')
        if(len(df2)>0):
            if (pd.Timestamp(start_time,tz='Europe/Zurich')>pd.Timestamp(df2.date.min(),tz='Europe/Zurich')):
                start_time=df2.date.max().to_pydatetime()
            else:
                end_time=df2.date.min().to_pydatetime()
    except:
        pass
    
    create_sqllite_table(table_name,engine)
    print 'reading from %s to %s'%(start_time,end_time) 
    #start_time=start_time.replace(tzinfo=None)
    start_time=pd.Timestamp(start_time,tz='Europe/Zurich')
    end_time=pd.Timestamp(end_time,tz='Europe/Zurich')
    time_list=list(pd.date_range(start_time,end_time,freq='6H').tz_convert('Europe/Zurich'))

    if(len(time_list)<=1):
        print 'time too short, enter time like yyyy-mm-dd-hh-mm-ss'
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
            if(len(df)>0):
                df.drop(cols_to_rem,axis=1,inplace=True)
                df.drop_duplicates(subset=['id'],inplace=True)
                df.to_sql(table_name,con=engine,if_exists='append',index=False)
            
        
    print('read from %s to %s'%(start_time,end_time))
    return True
    
    
def create_sqllite_table(name,db):
     
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
 
    
def main():
    '''
        run by python read_tweets.py table_name hashtag_name path_to_save start_time end_time
        or, to use default:  location, start and end time ; 
        use python read_tweets.py table_name 
    '''
    table_name=sys.argv[1]
    print table_name
    hashtag_name=sys.argv[2]
    print hashtag_name
    path='data.db'
    start_time='2013-01-01-16-00-00'
    end_time='2017-10-29-16-00-00'
    if(len(sys.argv)==4):
        path=sys.argv[3]
    if(len(sys.argv)>=5):
        start_time=sys.argv[4]
    if(len(sys.argv)>=6):
        end_time=sys.argv[5]
    result=False
    while(not result):
        try:
            result=read_tweets(table_name,hashtag_name,path,start_time,end_time)
            if not(type(result) == bool):
                result=False
        except:
            pass
                
    
if __name__=='__main__':
     main()
    
    
     
     
            
            
    