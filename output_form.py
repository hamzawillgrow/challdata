import pandas as pd
df=pd.read_csv('ytest_pred.csv', header=None)
# Original mapping
prdtypecode_mapping = {10: 0, 40: 1, 50: 2, 60: 3, 1140: 4, 1160: 5, 1180: 6, 1280: 7, 1281: 8, 1300: 9, 
                       1301: 10, 1302: 11, 1320: 12, 1560: 13, 1920: 14, 1940: 15, 2060: 16, 2220: 17, 
                       2280: 18, 2403: 19, 2462: 20, 2522: 21, 2582: 22, 2583: 23, 2585: 24, 2705: 25, 
                       2905: 26}

# Reverse the mapping
reverse_prdtypecode_mapping = {v: k for k, v in prdtypecode_mapping.items()}

# Apply the reversed mapping to ytrain
df.replace(reverse_prdtypecode_mapping, inplace=True)

print("Reversed ytrain:")
print(df)

df2=pd.read_csv('xtest.csv', header=None)
print(df)
df.iloc[:, 0] = df.iloc[:, 0].shift(-1) + 84916

#the last row has NaN value, so change it to the value of the previous row + 1
df.iloc[-1, 0] = df.iloc[-2, 0] + 1 


#add the headers [,prdtycode] to the dataframe
df.columns = ['','prdtypecode']

print(df2.shape)
print(df.shape)
print(df)
df.astype(int).to_csv('ytest_unmapped.csv', index=False,header=True)

