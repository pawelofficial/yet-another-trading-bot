col='close',window=tf,inplace=False)
    df[f'std-{tf}']=m.calculate_fun(df=df,fun_name='std'