import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math   

#data import
dat = []
start_year = 2013
num_element = 0
month = 6

custom_headers = [round(i * 0.1, 1) for i in range(-10, 11)]
df = pd.read_csv('results.csv', delimiter=';',  header=None, names=custom_headers)
df[1] = 0   #adding 0 in last column instead of NaN
date_blocks = df.iloc[:, 0].str.split('_').str[3]
date_blocks_split = date_blocks.apply(lambda x: pd.Series(list(str(x))))
# Rename the new columns
date_blocks_split.columns = [f'digit_{i}' for i in range(1, date_blocks_split.shape[1] + 1)]

# Generate a list of row numbers where digit_5 is 5
rows_with_digit_5_equal_5 = date_blocks_split[date_blocks_split['digit_6'] == str(month)].index.tolist()
selected_rows_df = df.iloc[rows_with_digit_5_equal_5]
numerical_data = selected_rows_df.iloc[:, 1:].to_numpy()

data_px = np.transpose(np.asarray(numerical_data)) #rotate matrix
epsilon = 1
data = data_px.astype(int)
# data = np.round(np.log(data_px + epsilon)).astype(int)
# data = np.asarray(data_px)
max_value = np.max(data)
num_element = np.shape(data)[1]
#ndvi range
ndvi = np.arange(-1, 1, 0.1).tolist()
decimal_places = 2
ndvi = np.round(ndvi, decimals=decimal_places)
#years values
years = list(range(start_year,start_year+num_element+1))  #make labels for x axis

#x,y labels
plt.imshow(data, cmap='YlGn', interpolation='nearest')

# Create colorbar
# plt.colorbar(im, ticks=np.linspace(0, max_value, max_value+1))     
plt.title("NDVI over years")
plt.xticks(range(len(years)),years, rotation=90)
plt.yticks(range(len(ndvi)),ndvi)
ax = plt.gca()
ax.invert_yaxis()
plt.colorbar(label="number of px")   
plt.show()