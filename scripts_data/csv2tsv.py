import pandas as pd

folder_pth = 'dataset'
csv_file_name = "0629.csv"
flag = "_variant"

train_set = pd.read_csv(f'{folder_pth}/{csv_file_name}.csv', sep=',', header=0)

train_df_bert = pd.DataFrame({
    'label': train_set['label2'],
    'text_a': train_set['text'].replace(r'\n', ' ', regex=True),
    #'text2': train_set['sentence2'].replace(r'\n', ' ', regex=True)

})
train_df_bert.to_csv(f'{folder_pth}/{csv_file_name}{flag}.tsv', sep='\t', index=False, header=True)
