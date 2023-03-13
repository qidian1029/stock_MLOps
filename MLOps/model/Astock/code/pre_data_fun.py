import numpy as np
import pandas as pd
import ast

Astock_path = "C:/Users/is_li/Desktop/code&data/Astock/data/"

def string_to_tuples_list(text):
    if text is np.nan or text =='[]':
        return []
    text = ''.join(text.split('], ['))
    tmp = eval(text.strip('[').strip(']'))
    if not isinstance(tmp[0],tuple):
        return [tmp]
    return list(tmp)

def mask(df):
  df = df.reset_index(drop = True)
  df['verb_mask'] = 0
  df['A0_mask'] = 0
  df['A1_mask'] = 0
  df['verb_mask'] = df['verb_mask'].astype('object')
  df['A0_mask'] = df['A0_mask'].astype('object')
  df['A1_mask'] = df['A1_mask'].astype('object')
  for index,row in df.iterrows():

    df.at[index,'stock_factors'] = [*map(float,df.loc[index,'stock_factors'])]
    AV_num = 0
    for k,col in enumerate(['verb','A0','A1']):
      masks = []
      for j in range(len(row['verbA0A1'])):
        mask = np.zeros(299)
        idx = []
        for v in row['verbA0A1'][j][k]:
          
          idx = idx + [int(i) for i in range(v[0],v[0]+v[1])]
        # idx = np.unique(idx).tolist()
        counter = Counter(idx)

        mask = [0 if counter[i]== 0 else 1/len(counter) for i in range(0,len(mask))]
        mask.insert(0,0)
        masks.append(mask)
      AV_num = len(masks)
      for i in range(10 - len(masks)):
        masks.append(np.zeros(300))
      while len(masks)>10:
        masks.pop()
      name = col+'_mask'
      df.at[index,name] = np.array(masks)
    if AV_num>10:
      AV_num=10
    df.loc[index,'AV_num'] = int(AV_num)
    df.AV_num = df.AV_num.astype('int')
    df.stock_factors = df.stock_factors.apply(np.array)
    return df

def Generating_mask(Astock_path,df_train,df_val,df_test,df_ood):

    df_train = df_train.drop(df_train.loc[df_train.verbA0A1.isna()].index)
    df_test = df_test.drop(df_test.loc[df_test.verbA0A1.isna()].index)
    df_val = df_val.drop(df_val.loc[df_val.verbA0A1.isna()].index)
    df_ood = df_ood.drop(df_ood.loc[df_ood.verbA0A1.isna()].index)

    df_train = df_train.drop(df_train.loc[df_train.verbA0A1=='[]'].index)
    df_test = df_test.drop(df_test.loc[df_test.verbA0A1=='[]'].index)
    df_val = df_val.drop(df_val.loc[df_val.verbA0A1=='[]'].index)
    df_ood = df_ood.drop(df_ood.loc[df_ood.verbA0A1=='[]'].index)


    for col in ['verb','A0','A1']:
        df_train[col] = df_train[col].apply(string_to_tuples_list)
        df_val[col] = df_val[col].apply(string_to_tuples_list)
        df_test[col] = df_test[col].apply(string_to_tuples_list)
        df_ood[col] = df_ood[col].apply(string_to_tuples_list)

    for col in ['stock_factors','verbA0A1']:
    # for col in ['verbA0A1']:
        df_train[col] = df_train[col].apply(ast.literal_eval)
        df_val[col] = df_val[col].apply(ast.literal_eval)
        df_test[col] = df_test[col].apply(ast.literal_eval)
        df_ood[col] = df_ood[col].apply(ast.literal_eval)

        df_train = mask(df_train)
        df_test = mask(df_test)
        df_val = mask(df_val)
        df_ood = mask(df_ood)

    return df_train,df_val,df_test,df_ood


