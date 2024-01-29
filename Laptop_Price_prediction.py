import pandas as pd

#Read the CSV file into a DataFrame

df = pd.read_csv('Old_laptop_Price.csv')

#Display the first few rows of the DataFrame

df.head()

#Exploratory Data Analysis (EDA)

df.info()
df.describe()

#Data Cleaning

#Handling missing values

null_values = df.isnull()
null_values

## Couting Null values

missing_values = null_values.sum()
missing_values

# Display columns with missing values

print("Columns with missing values:")
print(missing_values[missing_values>0])

# Drop rows with any missing values

df_drop_rows = df.dropna()

# Drop columns with any missing values

df_drop_rows = df.dropna(axis=1)

# Drop rows only if all columns have missing values

df_drop_rows_all = df.dropna(how='all')
df

#Data Visualization

import matplotlib.pyplot as plt

# Calculate counts for each brand
brand_counts = df['Brand'].value_counts()
brand_counts

# Create a pie chart for Brand distribution
import matplotlib.pyplot as plt
plt.figure(figsize = (8,8))
plt.pie(brand_counts,labels=brand_counts.index,autopct='%1.1f%%',startangle=140)
plt.title('Brand Distribution')
plt.axis('equal')
plt.show()

# Assuming 'Year of Release' as time and 'Actual Price (Rs)' as the variable of interest
#Line Chart
plt.figure(figsize = (10,6))
plt.plot(df['Year of Release'],df['Actual Price (Rs)'],marker='o',linestyle='-')
plt.title('Actual Price (Rs) over Years')
plt.xlabel('Year of Release')
plt.ylabel('Actual Price (Rs)')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate average Old Laptop Selling Price per Brand

avg_price_per_brand = df.groupby('Brand')['Old Laptop Selling Price (Rs)'].mean().sort_values(ascending=False)

# Create a bar plot for Average Selling Price per Brand

plt.figure(figsize=(10,6))
sns.barplot(x=avg_price_per_brand.index, y=avg_price_per_brand.values, palette='viridis')
plt.title('Average Old Laptop Selling Price per Brand')
plt.xlabel('Brand')
plt.ylabel('Average Selling Price (Rs)')
plt.xticks(rotation=45)
plt.show()

df1 = df.copy()

#Feature Engineering
#Adding new or modifying existing columns in the dataset for the machine learning models to understand the data more efficiently

df1.rename(columns={'Keyword Working':'Keyboard Working'})

df1['Touchpad Working']= ['Yes','Yes','Yes','No','Yes','Yes','Yes','No','Yes',
                          'Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes']

df1['Screen Touch']= ['No','No','No','No','No','Yes','Yes','No','No',
                      'Yes','No','No','No','Yes','Yes','Yes','Yes']
df1

df1['Touchpad Working']= df1['Touchpad Working'].map({'Yes':1,'No':0})
df1['Screen Touch']= df1['Screen Touch'].map({'Yes':1,'No':0})
df1

# Calculate depreciation percentage

df1['Depreciation (%)'] = ((df1['Actual Price (Rs)'] - df1['Old Laptop Selling Price (Rs)']) / df1['Actual Price (Rs)']) * 100

# Create a feature indicating if the selling price is high or low compared to the actual price

df1['Price Status'] = df1['Old Laptop Selling Price (Rs)'] > df['Actual Price (Rs)']
df1['Price Status'] = df1['Price Status'].map({True: 'High', False: 'Low'})

print("\nDataset after Feature Engineering:")
df1

#Statistical Analysis

# Summary statistics of numerical columns

summary_stats = df.describe()
summary_stats

#Machine Learning

df2 = df1.copy()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
# Label Encoding for categorical variables
label_encoder = LabelEncoder()
df2['Brand'] = label_encoder.fit_transform(df2['Brand'])
df2['Processor'] = label_encoder.fit_transform(df2['Processor'])
df2['Physical Condition'] = label_encoder.fit_transform(df2['Physical Condition'])
df2['Operating System'] = label_encoder.fit_transform(df2['Operating System'])

# Convert 'Storage (GB)' column to numeric by extracting numerical part
df2['Storage (GB)'] = df2['Storage (GB)'].str.extract('(\d+)').astype(float)
# Define features and target variable
features = ['RAM (GB)', 'Storage (GB)', 'Year of Release', 'Age (Years)', 'Screen Damage',
            'Charger Availability', 'Keyword Working', 'Battery Health (%)','Screen Touch','Touchpad Working']
target = 'Old Laptop Selling Price (Rs)'

X = df2[features]
y = df2[target]

# Train the linear regression model with the entire dataset
model = LinearRegression()
model.fit(X, y)

# Take user input
user_input = {
    'RAM (GB)': 8,
    'Storage (GB)': 256,
    'Year of Release': 2020,
    'Age (Years)': 2,
    'Screen Damage': 0,  # 0 for No
    'Charger Availability': 1,  # 1 for Yes
    'Keyword Working': 1,  # 1 for Yes
     'Battery Health (%)': 90,
    'Screen Touch': 1,
    'Touchpad Working': 1
}
# Predict the selling price for the user input
user_input_df = pd.DataFrame([user_input])
user_pred = model.predict(user_input_df[features])
print(f'Predicted Selling Price: {user_pred[0]}')

predicted_price = user_pred[0]

print(predicted_price)

df2

#Using DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor

# Define features and target variable
features = ['RAM (GB)', 'Storage (GB)', 'Year of Release', 'Age (Years)', 'Screen Damage',
            'Charger Availability', 'Keyword Working', 'Battery Health (%)','Screen Touch','Touchpad Working']

target = 'Old Laptop Selling Price (Rs)'

X = df2[features]
y = df2[target]

# Train the linear regression model with the entire dataset
model = DecisionTreeRegressor()
model.fit(X, y)

# Take user input
user_input = {
    'RAM (GB)': 8,
    'Storage (GB)': 256,
    'Year of Release': 2020,
    'Age (Years)': 2,
    'Screen Damage': 0,  # 0 for No
    'Charger Availability': 1,  # 1 for Yes
    'Keyword Working': 1,  # 1 for Yes
     'Battery Health (%)': 90,
    'Screen Touch': 0,
    'Touchpad Working': 0
}
# Predict the selling price for the user input
user_input_df = pd.DataFrame([user_input])
user_pred = model.predict(user_input_df[features])
print(f'Predicted Selling Price: {user_pred[0]}')

#Using RandomForestRegressor

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Define features and target variable
features = ['RAM (GB)', 'Storage (GB)', 'Year of Release', 'Age (Years)', 'Screen Damage',
            'Charger Availability', 'Keyword Working', 'Battery Health (%)','Screen Touch','Touchpad Working']

target = 'Old Laptop Selling Price (Rs)'

X = df2[features]
y = df2[target]

# Train the decision tree regressor model with the entire dataset
model = RandomForestRegressor(n_estimators = 3,random_state=42)
model.fit(X, y)

# Take user input
user_input = {
    'RAM (GB)': 8,
    'Storage (GB)': 256,
    'Year of Release': 2020,
    'Age (Years)': 2,
    'Screen Damage': 0,  # 0 for No
    'Charger Availability': 1,  # 1 for Yes
    'Keyword Working': 1,  # 1 for Yes
    'Battery Health (%)': 90,
    'Screen Touch': 1,
    'Touchpad Working': 1
}
# Predict the selling price for the user input
user_input_df = pd.DataFrame([user_input])
user_pred = model.predict(user_input_df[features])
print(f'Predicted Selling Price: {user_pred[0]}')
