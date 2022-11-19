from NNfuns import * 



if __name__=='__main__':
#    raw_df = read_raw_data_from_api()
#    raw_df = read_df_from_file(filename='BTC-USD2022-01-10_2022-01-18')
#    dump_df(df=raw_df,filename='')
#    agg_df=aggregate_df(df=raw_df)
#    dump_df(df=agg_df,filename='agg_df')
    agg_df=read_df_from_file(filename='agg_df_small')
    feature_df=make_features(agg_df)
    
    
    dump_df(df=feature_df,filename='feature_df')
    print(feature_df)
    


