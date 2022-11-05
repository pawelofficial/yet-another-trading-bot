from NNfuns import * 



if __name__=='__main__':
    raw_df = read_raw_data_from_api()
#    raw_df = read_raw_data_from_file()
    dump_raw_data(df=raw_df)
    agg_df=aggregate_df(df=raw_df)
    dump_raw_data(df=agg_df,filename='agg_df')
    
