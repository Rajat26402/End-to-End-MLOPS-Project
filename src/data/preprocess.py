import pandas as pd

def load_and_combine(true_path, fake_path):
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df['label'] = 1
    fake_df['label'] = 0

    df = pd.concat([true_df, fake_df]).reset_index(drop=True)
    # data=df.drop(['title','subject','date','text_len'], axis = 1)
    # data.isnull().sum() 
    return df
