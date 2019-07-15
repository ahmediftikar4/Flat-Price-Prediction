
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from scipy.stats import skew
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import joblib
warnings.filterwarnings('ignore')



# In[2]:


# Load train and Test set
train = pd.read_csv("./Connect/train.csv")
test = pd.read_csv("./Connect/test.csv")


# In[3]:


# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Now drop the 'Id' column since it's unnecessary for the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


# Checking Categorical Data
train.select_dtypes(include=['object']).columns


# In[7]:


# Checking Numerical Data
train.select_dtypes(include=['int64','float64']).columns


# In[8]:


cat = len(train.select_dtypes(include=['object']).columns)
num = len(train.select_dtypes(include=['int64','float64']).columns)


# In[9]:


# Correlation Matrix Heatmap
corrmat = train.corr()


# In[10]:


# Top 10 Heatmap
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)


# In[11]:


most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr


# In[12]:


# Combining Datasets
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)


# In[13]:


# Find Missing Ratio of Dataset
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data


# In[14]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data['GarageType'] = all_data['GarageType'].fillna('None')

    
all_data = all_data.drop(['Utilities'], axis=1)
all_data['NoF'] = all_data['NoF'].fillna("None")


# In[15]:


# Check if there are any missing values left
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[16]:


all_data['NoF'].describe()


# In[17]:


#NoF =The building class
all_data['NoF'] = all_data['NoF'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


# In[18]:


cols = ['NoF', 'OverallCond', 'PoolQC', 'Street']

# Process columns and apply LabelEncoder to categorical features
lbl = LabelEncoder()
lbl.fit([y for x in all_data[cols].get_values() for y in x])

# Saving the label encoder to pickle file
output = open('Encoder.pkl', 'wb')
pickle.dump(lbl, output)
output.close()

all_data[cols] = all_data[cols].apply(lbl.transform)


# In[19]:


# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

y_train = train.SalePrice.values


# In[20]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skewed Features' :skewed_feats})
skewness.head()


# In[21]:


skewness = skewness[abs(skewness) > 0.75]

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    all_data[feat] += 1


# In[22]:


labelencoder_dict = {}
onehotencoder_dict = {}
all_data_train = None
all_data_array = all_data.values

for i in range(0, all_data_array.shape[1]):
    if i in [1,2,3,4,5]:
        label_encoder = LabelEncoder()
        labelencoder_dict[i] = label_encoder
        feature = label_encoder.fit_transform(all_data_array[:,i])
        feature = feature.reshape(all_data_array.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        feature = onehot_encoder.fit_transform(feature)
        onehotencoder_dict[i] = onehot_encoder
    else:
        feature = all_data_array[:,i].reshape(all_data_array.shape[0], 1)
    if all_data_train is None:
        all_data_train = feature
    else:
        all_data_train = np.concatenate((all_data_train, feature), axis=1)

joblib.dump(labelencoder_dict, 'labelencoder_dict.joblib')
joblib.dump(onehotencoder_dict, 'onehotencoder_dict.joblib')


# In[23]:


train = all_data_train[:ntrain]
test = all_data_train[ntrain:]


# In[24]:


# Cross-validation with k-folds
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[25]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1,
                             random_state =7)


# In[26]:


score = rmsle_cv(model_xgb)


# In[27]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[28]:


model_xgb.fit(train, y_train)
joblib.dump(model_xgb, 'xgboost_model.joblib')
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))


# In[29]:


print('RMSLE score on train data:')
print(rmsle(y_train, xgb_train_pred*0.10 ))


# In[30]:


# Example
XGBoost = 1/(0.1177)


# In[31]:


ensemble = xgb_pred*XGBoost


# In[32]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)
print("The Sale Price for the test.csv file is updated successfully and stored in submission.csv file\n")

