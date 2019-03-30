
# Cancer Treatment

You can check all details about the competition from following link :
https://www.kaggle.com/c/msk-redefining-cancer-treatment

We are trying to distinguish the mutations that contribute to tumor growth from the neutral mutations. 

It's a **multiclass classification probelm**: we will classify genetic variations in to different cancer classes.


```python
# Load tools
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold 
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
```


```python
# Loading training_variants. Its a comma seperated file
data_variants = pd.read_csv('training/training_variants')
# Loading training_text dataset. This is seperated by ||
data_text =pd.read_csv("training/training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
```


```python
data_variants.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Gene</th>
      <th>Variation</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>FAM58A</td>
      <td>Truncating Mutations</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>CBL</td>
      <td>W802*</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>CBL</td>
      <td>Q249E</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



* ID : row id used to link the mutation to the clinical evidence
* Gene : the gene where this genetic mutation is located 
* Variation : the aminoacid change for this mutations 
* Class : class value 1-9, this genetic mutation has been classified on


```python
data_variants.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3321 entries, 0 to 3320
    Data columns (total 4 columns):
    ID           3321 non-null int64
    Gene         3321 non-null object
    Variation    3321 non-null object
    Class        3321 non-null int64
    dtypes: int64(2), object(2)
    memory usage: 103.9+ KB



```python
data_variants.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3321.000000</td>
      <td>3321.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1660.000000</td>
      <td>4.365854</td>
    </tr>
    <tr>
      <th>std</th>
      <td>958.834449</td>
      <td>2.309781</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>830.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1660.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2490.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3320.000000</td>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_variants.shape
```




    (3321, 4)




```python
data_variants.columns
```




    Index(['ID', 'Gene', 'Variation', 'Class'], dtype='object')




```python
data_text.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>TEXT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Cyclin-dependent kinases (CDKs) regulate a var...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Abstract Background  Non-small cell lung canc...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Abstract Background  Non-small cell lung canc...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_text.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3321 entries, 0 to 3320
    Data columns (total 2 columns):
    ID      3321 non-null int64
    TEXT    3316 non-null object
    dtypes: int64(1), object(1)
    memory usage: 52.0+ KB



```python
data_text.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3321.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1660.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>958.834449</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>830.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1660.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2490.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3320.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_text.columns
```




    Index(['ID', 'TEXT'], dtype='object')




```python
data_text.shape
```




    (3321, 2)



 * data_variants (ID, Gene, Variations, Class)
 * data_text(ID, text)


```python
data_variants.Class.unique()
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])



## NLP on text


```python
import nltk
nltk.download('stopwords')
# remove common meaningless words
stop_words = set(stopwords.words('english'))
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/luoqiaoen/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
def data_text_preprocess(total_text, ind, col):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from text
            if not word in stop_words:
                string += word + " "
        
        data_text[col][ind] = string
```


```python
for index, row in data_text.iterrows():
    if type(row['TEXT']) is str:
        data_text_preprocess(row['TEXT'], index, 'TEXT')
```


```python
#merging both gene_variations and text data based on ID like INNER JOIN!
result = pd.merge(data_variants, data_text,on='ID', how='left')
result.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Gene</th>
      <th>Variation</th>
      <th>Class</th>
      <th>TEXT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>FAM58A</td>
      <td>Truncating Mutations</td>
      <td>1</td>
      <td>cyclin dependent kinases cdks regulate variety...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>CBL</td>
      <td>W802*</td>
      <td>2</td>
      <td>abstract background non small cell lung cancer...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>CBL</td>
      <td>Q249E</td>
      <td>2</td>
      <td>abstract background non small cell lung cancer...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>CBL</td>
      <td>N454D</td>
      <td>3</td>
      <td>recent evidence demonstrated acquired uniparen...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>CBL</td>
      <td>L399V</td>
      <td>4</td>
      <td>oncogenic mutations monomeric casitas b lineag...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# missing data can mess up a lot of things, need to check
result[result.isnull().any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Gene</th>
      <th>Variation</th>
      <th>Class</th>
      <th>TEXT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1109</th>
      <td>1109</td>
      <td>FANCA</td>
      <td>S1088F</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1277</th>
      <td>1277</td>
      <td>ARID5B</td>
      <td>Truncating Mutations</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1407</th>
      <td>1407</td>
      <td>FGFR3</td>
      <td>K508M</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1639</th>
      <td>1639</td>
      <td>FLT1</td>
      <td>Amplification</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2755</th>
      <td>2755</td>
      <td>BRAF</td>
      <td>G596C</td>
      <td>7</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we can fill the NaN as Gene + Variation
result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']
```


```python
# check again
result[result.isnull().any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Gene</th>
      <th>Variation</th>
      <th>Class</th>
      <th>TEXT</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



# Create Trainig, Testing and Validation Data


```python
# replace all space with _
y_true = result['Class'].values
result.Gene      = result.Gene.str.replace('\s+', '_')
result.Variation = result.Variation.str.replace('\s+', '_')
```


```python
# Splitting the data into train and test set 
X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2)
# split the train data now into train validation and cross validation
train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
```


```python
print('Number of data points in train data:', train_df.shape[0])
print('Number of data points in test data:', test_df.shape[0])
print('Number of data points in cross validation data:', cv_df.shape[0])
```

    Number of data points in train data: 2124
    Number of data points in test data: 665
    Number of data points in cross validation data: 532



```python
train_class_distribution = train_df['Class'].value_counts().sort_index()
test_class_distribution = test_df['Class'].value_counts().sort_index()
cv_class_distribution = cv_df['Class'].value_counts().sort_index()
```


```python
train_class_distribution
```




    1    363
    2    289
    3     57
    4    439
    5    155
    6    176
    7    609
    8     12
    9     24
    Name: Class, dtype: int64




```python
test_class_distribution
```




    1    114
    2     91
    3     18
    4    137
    5     48
    6     55
    7    191
    8      4
    9      7
    Name: Class, dtype: int64




```python
cv_class_distribution
```




    1     91
    2     72
    3     14
    4    110
    5     39
    6     44
    7    153
    8      3
    9      6
    Name: Class, dtype: int64



### Visualization of Training Data


```python
train_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel(' Number of Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_33_0.png)



```python
sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3), '%)')
```

    Number of data points in class 7 : 609 ( 28.672 %)
    Number of data points in class 4 : 439 ( 20.669 %)
    Number of data points in class 1 : 363 ( 17.09 %)
    Number of data points in class 2 : 289 ( 13.606 %)
    Number of data points in class 6 : 176 ( 8.286 %)
    Number of data points in class 5 : 155 ( 7.298 %)
    Number of data points in class 3 : 57 ( 2.684 %)
    Number of data points in class 9 : 24 ( 1.13 %)
    Number of data points in class 8 : 12 ( 0.565 %)


### Visualizing Test Data


```python
test_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Number of Data points per Class')
plt.title('Distribution of yi in test data')
plt.grid()
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_36_0.png)



```python
sorted_yi = np.argsort(-test_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_df.shape[0]*100), 3), '%)')
```

    Number of data points in class 7 : 191 ( 28.722 %)
    Number of data points in class 4 : 137 ( 20.602 %)
    Number of data points in class 1 : 114 ( 17.143 %)
    Number of data points in class 2 : 91 ( 13.684 %)
    Number of data points in class 6 : 55 ( 8.271 %)
    Number of data points in class 5 : 48 ( 7.218 %)
    Number of data points in class 3 : 18 ( 2.707 %)
    Number of data points in class 9 : 7 ( 1.053 %)
    Number of data points in class 8 : 4 ( 0.602 %)


### Visualizing Cross Validation Data


```python
my_colors = 'rgbkymc'
cv_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in cross validation data')
plt.grid()
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_39_0.png)



```python
sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',cv_class_distribution.values[i], '(', np.round((cv_class_distribution.values[i]/cv_df.shape[0]*100), 3), '%)')
```

    Number of data points in class 7 : 153 ( 28.759 %)
    Number of data points in class 4 : 110 ( 20.677 %)
    Number of data points in class 1 : 91 ( 17.105 %)
    Number of data points in class 2 : 72 ( 13.534 %)
    Number of data points in class 6 : 44 ( 8.271 %)
    Number of data points in class 5 : 39 ( 7.331 %)
    Number of data points in class 3 : 14 ( 2.632 %)
    Number of data points in class 9 : 6 ( 1.128 %)
    Number of data points in class 8 : 3 ( 0.564 %)


# We build a random model for comparison later
Any later models need to outperform this random model


```python
test_data_len = test_df.shape[0]
cv_data_len = cv_df.shape[0]

cv_predicted_y = np.zeros((cv_data_len,9))
for i in range(cv_data_len):
    rand_probs = np.random.rand(1,9)
    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))
#Log loss is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p)).
```

    Log loss on Cross Validation Data using Random Model 2.3882939267546517



```python
test_predicted_y = np.zeros((test_data_len,9))
for i in range(test_data_len):
    rand_probs = np.random.rand(1,9)
    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))
```

    Log loss on Test Data using Random Model 2.508350930572329



```python
predicted_y =np.argmax(test_predicted_y, axis=1)
```


```python
predicted_y + 1 # show the most probable class of cancer
```




    array([3, 8, 5, 9, 1, 8, 3, 8, 1, 3, 5, 8, 1, 4, 5, 3, 4, 9, 5, 9, 7, 9,
           3, 7, 1, 5, 2, 5, 9, 3, 6, 6, 2, 1, 3, 1, 5, 2, 7, 3, 7, 4, 9, 3,
           3, 4, 6, 8, 3, 8, 9, 1, 8, 1, 3, 3, 6, 6, 1, 4, 4, 9, 1, 3, 9, 4,
           5, 7, 6, 9, 6, 9, 1, 5, 9, 7, 1, 2, 2, 2, 7, 6, 5, 8, 9, 3, 7, 5,
           2, 4, 1, 9, 8, 3, 9, 3, 6, 1, 5, 2, 4, 2, 2, 5, 9, 2, 2, 3, 5, 5,
           7, 3, 6, 6, 5, 2, 6, 6, 4, 8, 7, 5, 2, 2, 7, 4, 9, 5, 1, 9, 3, 3,
           7, 4, 6, 3, 3, 8, 9, 4, 2, 6, 8, 4, 3, 4, 3, 4, 6, 5, 2, 6, 5, 6,
           7, 2, 2, 6, 8, 1, 4, 2, 8, 2, 6, 7, 6, 9, 4, 2, 1, 4, 6, 3, 8, 7,
           7, 9, 1, 9, 4, 6, 6, 1, 3, 6, 7, 1, 8, 8, 8, 4, 9, 6, 8, 6, 7, 4,
           9, 3, 5, 6, 2, 5, 3, 4, 6, 5, 7, 5, 3, 7, 6, 5, 5, 1, 7, 4, 9, 1,
           6, 7, 1, 2, 2, 4, 1, 7, 8, 4, 4, 4, 7, 3, 1, 8, 3, 1, 8, 4, 9, 1,
           7, 9, 7, 1, 6, 1, 5, 4, 9, 3, 7, 4, 8, 2, 4, 8, 5, 8, 7, 7, 6, 4,
           8, 8, 5, 2, 4, 8, 5, 7, 6, 1, 1, 5, 1, 9, 2, 5, 4, 2, 7, 6, 5, 3,
           1, 2, 6, 2, 9, 7, 9, 2, 6, 3, 3, 3, 7, 6, 6, 2, 7, 2, 2, 5, 2, 4,
           7, 1, 6, 2, 1, 3, 2, 8, 9, 8, 2, 6, 1, 9, 1, 4, 6, 6, 9, 3, 4, 7,
           9, 4, 8, 8, 1, 8, 5, 5, 3, 1, 8, 8, 1, 4, 9, 5, 9, 7, 5, 9, 5, 3,
           8, 1, 3, 3, 9, 3, 9, 4, 2, 8, 8, 9, 7, 8, 8, 5, 7, 7, 2, 4, 1, 3,
           2, 9, 9, 4, 9, 8, 9, 8, 9, 7, 9, 6, 2, 5, 9, 1, 6, 3, 7, 2, 5, 1,
           5, 4, 8, 4, 1, 4, 3, 9, 2, 6, 9, 1, 1, 1, 4, 3, 9, 4, 5, 7, 9, 7,
           8, 7, 5, 3, 3, 3, 3, 2, 7, 9, 9, 4, 4, 3, 5, 6, 2, 9, 8, 4, 5, 1,
           5, 3, 4, 4, 9, 1, 8, 9, 4, 8, 4, 4, 6, 5, 9, 6, 1, 9, 1, 7, 9, 7,
           4, 9, 6, 7, 3, 3, 9, 6, 2, 6, 1, 8, 1, 4, 6, 7, 3, 7, 6, 8, 9, 2,
           9, 2, 5, 3, 1, 4, 9, 1, 4, 7, 4, 3, 2, 7, 1, 3, 4, 5, 7, 8, 4, 2,
           8, 4, 7, 4, 6, 8, 3, 8, 7, 9, 3, 1, 4, 4, 2, 7, 2, 1, 8, 3, 9, 9,
           9, 6, 5, 2, 4, 2, 4, 6, 6, 7, 1, 2, 7, 3, 7, 9, 7, 7, 2, 9, 4, 4,
           9, 5, 7, 7, 5, 3, 2, 3, 3, 2, 7, 9, 4, 4, 3, 6, 6, 5, 8, 5, 3, 8,
           6, 5, 3, 1, 6, 5, 7, 2, 1, 1, 2, 4, 1, 3, 8, 6, 2, 1, 1, 3, 8, 4,
           6, 3, 9, 2, 4, 1, 7, 1, 6, 8, 5, 8, 5, 5, 2, 1, 7, 3, 5, 7, 4, 5,
           2, 1, 5, 3, 2, 8, 9, 8, 8, 2, 7, 3, 3, 8, 8, 9, 7, 8, 3, 9, 8, 3,
           3, 8, 8, 2, 1, 1, 3, 8, 4, 9, 4, 5, 8, 3, 5, 8, 2, 4, 4, 7, 9, 9,
           8, 5, 5, 4, 4])



#### Confusion Matrix


```python
C = confusion_matrix(y_test, predicted_y)
```


```python
labels = [1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(20,7))
sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_48_0.png)


#### Precision Matrix


```python
B =(C/C.sum(axis=0))

plt.figure(figsize=(20,7))
sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_50_0.png)


#### Recall Martrix


```python
A =(((C.T)/(C.sum(axis=1))).T)

plt.figure(figsize=(20,7))
sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_52_0.png)


## Evaluating Gene Column


```python
unique_genes = train_df['Gene'].value_counts()
print('Number of Unique Genes :', unique_genes.shape[0])
# the top 10 genes that occured most
print(unique_genes.head(10))
```

    Number of Unique Genes : 234
    BRCA1     176
    TP53      107
    EGFR       88
    PTEN       78
    BRCA2      75
    KIT        65
    ERBB2      46
    BRAF       44
    ALK        42
    PIK3CA     40
    Name: Gene, dtype: int64



```python
s = sum(unique_genes.values);
h = unique_genes.values/s;
c = np.cumsum(h)
plt.plot(c,label='Cumulative distribution of Genes')
plt.grid()
plt.legend()
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_55_0.png)


We need to convert categorical variable into something machine can make sense of, we can use:

1. ***One-hot encoding*** 
2. ***Response Encoding*** (Mean imputation) 



```python
# one-hot encoding of Gene feature.
gene_vectorizer = CountVectorizer()
train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])
```


```python
train_gene_feature_onehotCoding.shape
# 2124 data points, 233 unique genes
```




    (2124, 233)




```python
#column names after one-hot encoding for Gene column
gene_vectorizer.get_feature_names()
```




    ['abl1',
     'acvr1',
     'ago2',
     'akt1',
     'akt2',
     'akt3',
     'alk',
     'apc',
     'ar',
     'araf',
     'arid1a',
     'arid1b',
     'arid2',
     'arid5b',
     'asxl1',
     'asxl2',
     'atm',
     'atrx',
     'aurka',
     'aurkb',
     'axl',
     'b2m',
     'bap1',
     'bcl10',
     'bcor',
     'braf',
     'brca1',
     'brca2',
     'brip1',
     'btk',
     'card11',
     'carm1',
     'casp8',
     'cbl',
     'ccnd1',
     'ccnd2',
     'ccnd3',
     'ccne1',
     'cdh1',
     'cdk12',
     'cdk4',
     'cdk6',
     'cdk8',
     'cdkn1a',
     'cdkn1b',
     'cdkn2a',
     'cdkn2b',
     'cdkn2c',
     'cebpa',
     'chek2',
     'cic',
     'crebbp',
     'ctcf',
     'ctla4',
     'ctnnb1',
     'ddr2',
     'dicer1',
     'dnmt3a',
     'dnmt3b',
     'dusp4',
     'egfr',
     'eif1ax',
     'elf3',
     'ep300',
     'epas1',
     'epcam',
     'erbb2',
     'erbb3',
     'erbb4',
     'ercc2',
     'ercc3',
     'ercc4',
     'erg',
     'errfi1',
     'esr1',
     'etv1',
     'etv6',
     'ewsr1',
     'ezh2',
     'fam58a',
     'fanca',
     'fancc',
     'fat1',
     'fbxw7',
     'fgf19',
     'fgf3',
     'fgfr1',
     'fgfr2',
     'fgfr3',
     'fgfr4',
     'flt1',
     'flt3',
     'foxa1',
     'foxp1',
     'gata3',
     'gli1',
     'gna11',
     'gnaq',
     'gnas',
     'h3f3a',
     'hist1h1c',
     'hla',
     'hnf1a',
     'hras',
     'idh1',
     'idh2',
     'igf1r',
     'ikbke',
     'ikzf1',
     'il7r',
     'inpp4b',
     'jak1',
     'jak2',
     'kdm5c',
     'kdm6a',
     'kdr',
     'keap1',
     'kit',
     'klf4',
     'kmt2a',
     'kmt2b',
     'kmt2d',
     'knstrn',
     'kras',
     'lats1',
     'map2k1',
     'map2k2',
     'map2k4',
     'map3k1',
     'mapk1',
     'med12',
     'mef2b',
     'met',
     'mlh1',
     'mpl',
     'msh2',
     'msh6',
     'mtor',
     'myc',
     'myd88',
     'nf1',
     'nf2',
     'nfe2l2',
     'nfkbia',
     'nkx2',
     'notch1',
     'notch2',
     'npm1',
     'nras',
     'nsd1',
     'ntrk1',
     'ntrk2',
     'ntrk3',
     'nup93',
     'pak1',
     'pbrm1',
     'pdgfra',
     'pdgfrb',
     'pik3ca',
     'pik3cb',
     'pik3cd',
     'pik3r1',
     'pik3r2',
     'pim1',
     'pms2',
     'pole',
     'ppm1d',
     'ppp2r1a',
     'ppp6c',
     'prdm1',
     'ptch1',
     'pten',
     'ptpn11',
     'ptprd',
     'ptprt',
     'rab35',
     'rac1',
     'rad50',
     'rad51b',
     'rad51c',
     'rad51d',
     'rad54l',
     'raf1',
     'rasa1',
     'rb1',
     'rbm10',
     'ret',
     'rheb',
     'rhoa',
     'rictor',
     'rit1',
     'rnf43',
     'ros1',
     'rras2',
     'runx1',
     'rxra',
     'rybp',
     'sdhc',
     'setd2',
     'sf3b1',
     'smad2',
     'smad3',
     'smad4',
     'smarca4',
     'smarcb1',
     'smo',
     'sos1',
     'sox9',
     'spop',
     'src',
     'srsf2',
     'stat3',
     'stk11',
     'tcf3',
     'tcf7l2',
     'tert',
     'tet1',
     'tet2',
     'tgfbr1',
     'tgfbr2',
     'tmprss2',
     'tp53',
     'tp53bp1',
     'tsc1',
     'tsc2',
     'u2af1',
     'vegfa',
     'vhl',
     'whsc1',
     'whsc1l1',
     'xpo1',
     'xrcc2',
     'yap1']




```python
# code for response coding with Laplace smoothing.
# alpha : used for laplace smoothing
# feature: ['gene', 'variation']
# df: ['train_df', 'test_df', 'cv_df']
# algorithm
# ----------
# Consider all unique values and the number of occurances of given feature in train data dataframe
# build a vector (1*9) , the first element = (number of times it occured in class1 + 10*alpha / number of time it occurred in total data+90*alpha)
# gv_dict is like a look up table, for every gene it store a (1*9) representation of it
# for a value of feature in df:
# if it is in train data:
# we add the vector that was stored in 'gv_dict' look up table to 'gv_fea'
# if it is not there is train:
# we add [1/9, 1/9, 1/9, 1/9,1/9, 1/9, 1/9, 1/9, 1/9] to 'gv_fea'
# return 'gv_fea'
# ----------------------

# get_gv_fea_dict: Get Gene varaition Feature Dict
def get_gv_fea_dict(alpha, feature, df):
    # value_count: it contains a dict like
    # print(train_df['Gene'].value_counts())
    # output:
    #        {BRCA1      174
    #         TP53       106
    #         EGFR        86
    #         BRCA2       75
    #         PTEN        69
    #         KIT         61
    #         BRAF        60
    #         ERBB2       47
    #         PDGFRA      46
    #         ...}
    # print(train_df['Variation'].value_counts())
    # output:
    # {
    # Truncating_Mutations                     63
    # Deletion                                 43
    # Amplification                            43
    # Fusions                                  22
    # Overexpression                            3
    # E17K                                      3
    # Q61L                                      3
    # S222D                                     2
    # P130S                                     2
    # ...
    # }
    value_count = train_df[feature].value_counts()
    
    # gv_dict : Gene Variation Dict, which contains the probability array for each gene/variation
    gv_dict = dict()
    
    # denominator will contain the number of time that particular feature occured in whole data
    for i, denominator in value_count.items():
        # vec will contain (p(yi==1/Gi) probability of gene/variation belongs to perticular class
        # vec is 9 diamensional vector
        vec = []
        for k in range(1,10):
            # print(train_df.loc[(train_df['Class']==1) & (train_df['Gene']=='BRCA1')])
            #         ID   Gene             Variation  Class  
            # 2470  2470  BRCA1                S1715C      1   
            # 2486  2486  BRCA1                S1841R      1   
            # 2614  2614  BRCA1                   M1R      1   
            # 2432  2432  BRCA1                L1657P      1   
            # 2567  2567  BRCA1                T1685A      1   
            # 2583  2583  BRCA1                E1660G      1   
            # 2634  2634  BRCA1                W1718L      1   
            # cls_cnt.shape[0] will return the number of rows

            cls_cnt = train_df.loc[(train_df['Class']==k) & (train_df[feature]==i)]
            
            # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data
            vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))

        # we are adding the gene/variation to the dict as key and vec as value
        gv_dict[i]=vec
    return gv_dict

# Get Gene variation feature
def get_gv_feature(alpha, feature, df):
    # print(gv_dict)
    #     {'BRCA1': [0.20075757575757575, 0.03787878787878788, 0.068181818181818177, 0.13636363636363635, 0.25, 0.19318181818181818, 0.03787878787878788, 0.03787878787878788, 0.03787878787878788], 
    #      'TP53': [0.32142857142857145, 0.061224489795918366, 0.061224489795918366, 0.27040816326530615, 0.061224489795918366, 0.066326530612244902, 0.051020408163265307, 0.051020408163265307, 0.056122448979591837], 
    #      'EGFR': [0.056818181818181816, 0.21590909090909091, 0.0625, 0.068181818181818177, 0.068181818181818177, 0.0625, 0.34659090909090912, 0.0625, 0.056818181818181816], 
    #      'BRCA2': [0.13333333333333333, 0.060606060606060608, 0.060606060606060608, 0.078787878787878782, 0.1393939393939394, 0.34545454545454546, 0.060606060606060608, 0.060606060606060608, 0.060606060606060608], 
    #      'PTEN': [0.069182389937106917, 0.062893081761006289, 0.069182389937106917, 0.46540880503144655, 0.075471698113207544, 0.062893081761006289, 0.069182389937106917, 0.062893081761006289, 0.062893081761006289], 
    #      'KIT': [0.066225165562913912, 0.25165562913907286, 0.072847682119205295, 0.072847682119205295, 0.066225165562913912, 0.066225165562913912, 0.27152317880794702, 0.066225165562913912, 0.066225165562913912], 
    #      'BRAF': [0.066666666666666666, 0.17999999999999999, 0.073333333333333334, 0.073333333333333334, 0.093333333333333338, 0.080000000000000002, 0.29999999999999999, 0.066666666666666666, 0.066666666666666666],
    #      ...
    #     }
    gv_dict = get_gv_fea_dict(alpha, feature, df)
    # value_count is similar in get_gv_fea_dict
    value_count = train_df[feature].value_counts()
    
    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data
    gv_fea = []
    # for every feature values in the given data frame we will check if it is there in the train data then we will add the feature to gv_fea
    # if not we will add [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] to gv_fea
    for index, row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_fea.append(gv_dict[row[feature]])
        else:
            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])
#             gv_fea.append([-1,-1,-1,-1,-1,-1,-1,-1,-1])
    return gv_fea
```


```python
#response-coding of the Gene feature
# alpha is used for laplace smoothing
alpha = 1
# train gene feature
train_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", train_df))
# test gene feature
test_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", test_df))
# cross validation gene feature
cv_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", cv_df))
```


```python
train_gene_feature_responseCoding.shape
```




    (2124, 9)



Now, question is how good is Gene column feature to predict my 9 classes. One idea could be that we will build model having only gene column with one hot encoder with simple model like Logistic regression. If log loss with only one column Gene comes out to be better than random model, than this feature is important.


```python
# We need a hyperparemeter for SGD classifier.
alpha = [10 ** x for x in range(-5, 1)]
cv_log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_gene_feature_onehotCoding, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_gene_feature_onehotCoding, y_train)
    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    # alpha controls the size of regularization
```

    For values of alpha =  1e-05 The log loss is: 1.37570911471901
    For values of alpha =  0.0001 The log loss is: 1.1938384305915297
    For values of alpha =  0.001 The log loss is: 1.2166268610595454
    For values of alpha =  0.01 The log loss is: 1.3333931710508573
    For values of alpha =  0.1 The log loss is: 1.4162342964080283
    For values of alpha =  1 The log loss is: 1.4545687180081939



```python
fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_65_0.png)


best value of alpha =  0.0001 The log loss is: 1.200875750527778


```python
# Lets use best alpha value as we can see from above graph and compute log loss
best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_gene_feature_onehotCoding, y_train)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_gene_feature_onehotCoding, y_train)

predict_y = sig_clf.predict_proba(train_gene_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

```

    For values of best alpha =  0.0001 The train log loss is: 1.0433411064483948
    For values of best alpha =  0.0001 The cross validation log loss is: 1.1938384305915297
    For values of best alpha =  0.0001 The test log loss is: 1.2035719688982172


***We check how many values are overlapping between train, test or between CV and train***


```python
test_coverage=test_df[test_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]
cv_coverage=cv_df[cv_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]
```


```python
print('1. In test data',test_coverage, 'out of',test_df.shape[0], ":",(test_coverage/test_df.shape[0])*100)
print('2. In cross validation data',cv_coverage, 'out of ',cv_df.shape[0],":" ,(cv_coverage/cv_df.shape[0])*100)
```

    1. In test data 639 out of 665 : 96.09022556390977
    2. In cross validation data 519 out of  532 : 97.55639097744361


log loss is reduced by just doing a logistic regression on Gene column alone.

## Evaluating Variation Column
We will do similar stuff here with Variation


```python
unique_variations = train_df['Variation'].value_counts()
print('Number of Unique Variations :', unique_variations.shape[0])
# the top 10 variations that occured most
print(unique_variations.head(10))
```

    Number of Unique Variations : 1930
    Truncating_Mutations    57
    Amplification           45
    Deletion                43
    Fusions                 21
    G12V                     4
    T58I                     3
    Q61L                     3
    Q61H                     3
    TMPRSS2-ETV1_Fusion      2
    Y42C                     2
    Name: Variation, dtype: int64



```python
s = sum(unique_variations.values);
h = unique_variations.values/s;
c = np.cumsum(h)
print(c)
plt.plot(c,label='Cumulative distribution of Variations')
plt.grid()
plt.legend()
plt.show()
```

    [0.02683616 0.0480226  0.06826742 ... 0.99905838 0.99952919 1.        ]



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_73_1.png)



```python
# one-hot encoding of variation feature.
variation_vectorizer = CountVectorizer()
train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])
cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])
```


```python
train_variation_feature_onehotCoding.shape
```




    (2124, 1959)




```python
# alpha is used for laplace smoothing
alpha = 1
# train gene feature
train_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", train_df))
# test gene feature
test_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", test_df))
# cross validation gene feature
cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", cv_df))
```


```python
train_variation_feature_responseCoding.shape
```




    (2124, 9)




```python
alpha = [10 ** x for x in range(-5, 1)]
cv_log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_variation_feature_onehotCoding, y_train)
    
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_variation_feature_onehotCoding, y_train)
    predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)
    
    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
```

    For values of alpha =  1e-05 The log loss is: 1.7002163299812445
    For values of alpha =  0.0001 The log loss is: 1.6826976034802008
    For values of alpha =  0.001 The log loss is: 1.684356893560967
    For values of alpha =  0.01 The log loss is: 1.70070140340603
    For values of alpha =  0.1 The log loss is: 1.7103940218801863
    For values of alpha =  1 The log loss is: 1.7114371279209568



```python
fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_79_0.png)



```python
best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_variation_feature_onehotCoding, y_train)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_variation_feature_onehotCoding, y_train)

predict_y = sig_clf.predict_proba(train_variation_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
```

    For values of best alpha =  0.0001 The train log loss is: 0.7712566220843157
    For values of best alpha =  0.0001 The cross validation log loss is: 1.6826976034802008
    For values of best alpha =  0.0001 The test log loss is: 1.707045504122212



```python
test_coverage=test_df[test_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]
cv_coverage=cv_df[cv_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]
print('1. In test data',test_coverage, 'out of',test_df.shape[0], ":",(test_coverage/test_df.shape[0])*100)
print('2. In cross validation data',cv_coverage, 'out of ',cv_df.shape[0],":" ,(cv_coverage/cv_df.shape[0])*100)
```

    1. In test data 68 out of 665 : 10.225563909774436
    2. In cross validation data 59 out of  532 : 11.090225563909774


Seems like the Gene column is slightly more important? We can't be sure yet, move on to Text.

## Evaluating Text column

More NLP incoming.


```python
def extract_dictionary_paddle(cls_text):
    dictionary = defaultdict(int)
    for index, row in cls_text.iterrows():
        for word in row['TEXT'].split():
            dictionary[word] +=1
    return dictionary
```


```python
import math
#https://stackoverflow.com/a/1602964
def get_text_responsecoding(df):
    text_feature_responseCoding = np.zeros((df.shape[0],9))
    for i in range(0,9):
        row_index = 0
        for index, row in df.iterrows():
            sum_prob = 0
            for word in row['TEXT'].split():
                sum_prob += math.log(((dict_list[i].get(word,0)+10 )/(total_dict.get(word,0)+90)))
            text_feature_responseCoding[row_index][i] = math.exp(sum_prob/len(row['TEXT'].split()))
            row_index += 1
    return text_feature_responseCoding
```


```python
# building a CountVectorizer with all the words that occured minimum 3 times in train data
text_vectorizer = CountVectorizer(min_df=3)
train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])
# getting all the feature names (words)
train_text_features= text_vectorizer.get_feature_names()

# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector
train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1

# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured
text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))


print("Total number of unique words in train data :", len(train_text_features))
```

    Total number of unique words in train data : 52583



```python
dict_list = []
# dict_list =[] contains 9 dictoinaries each corresponds to a class
for i in range(1,10):
    cls_text = train_df[train_df['Class']==i]
    # build a word dict based on the words in that class
    dict_list.append(extract_dictionary_paddle(cls_text))
    # append it to dict_list

# dict_list[i] is build on i'th  class text data
# total_dict is buid on whole training text data
total_dict = extract_dictionary_paddle(train_df)


confuse_array = []
for i in train_text_features:
    ratios = []
    max_val = -1
    for j in range(0,9):
        ratios.append((dict_list[j][i]+10 )/(total_dict[i]+90))
    confuse_array.append(ratios)
confuse_array = np.array(confuse_array)
```


```python
#response coding of text features
train_text_feature_responseCoding  = get_text_responsecoding(train_df)
test_text_feature_responseCoding  = get_text_responsecoding(test_df)
cv_text_feature_responseCoding  = get_text_responsecoding(cv_df)
```


```python
# https://stackoverflow.com/a/16202486
# we convert each row values such that they sum to 1  
train_text_feature_responseCoding = (train_text_feature_responseCoding.T/train_text_feature_responseCoding.sum(axis=1)).T
test_text_feature_responseCoding = (test_text_feature_responseCoding.T/test_text_feature_responseCoding.sum(axis=1)).T
cv_text_feature_responseCoding = (cv_text_feature_responseCoding.T/cv_text_feature_responseCoding.sum(axis=1)).T
```


```python
# don't forget to normalize every feature
train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)

# we use the same vectorizer that was trained on train data
test_text_feature_onehotCoding = text_vectorizer.transform(test_df['TEXT'])
# don't forget to normalize every feature
test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)

# we use the same vectorizer that was trained on train data
cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['TEXT'])
# don't forget to normalize every feature
cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)
```


```python
#https://stackoverflow.com/a/2258273/4084039
sorted_text_fea_dict = dict(sorted(text_fea_dict.items(), key=lambda x: x[1] , reverse=True))
sorted_text_occur = np.array(list(sorted_text_fea_dict.values()))
```


```python
# Number of words for a given frequency.
print(Counter(sorted_text_occur))
```

    Counter({3: 5195, 4: 3820, 6: 2620, 5: 2556, 8: 2223, 9: 2094, 7: 1784, 10: 1234, 12: 1161, 11: 1122, 16: 991, 15: 866, 18: 847, 13: 840, 14: 821, 20: 620, 17: 588, 24: 550, 21: 494, 36: 481, 26: 468, 19: 460, 22: 413, 23: 407, 28: 400, 25: 367, 27: 355, 30: 336, 32: 319, 34: 312, 50: 303, 29: 289, 33: 288, 31: 258, 35: 252, 37: 232, 40: 218, 38: 211, 39: 195, 48: 187, 41: 183, 45: 180, 42: 179, 43: 170, 46: 167, 52: 160, 56: 158, 68: 148, 51: 148, 47: 146, 44: 145, 49: 135, 53: 129, 55: 127, 54: 124, 57: 122, 72: 121, 62: 116, 61: 113, 60: 113, 63: 107, 64: 106, 65: 102, 66: 101, 81: 99, 58: 96, 78: 94, 59: 93, 71: 91, 96: 89, 75: 89, 69: 89, 67: 88, 73: 83, 76: 81, 70: 80, 74: 79, 88: 77, 77: 77, 80: 76, 100: 70, 84: 68, 79: 68, 82: 66, 94: 65, 91: 65, 83: 63, 87: 62, 93: 61, 90: 61, 97: 60, 104: 58, 89: 58, 95: 56, 99: 55, 86: 55, 98: 54, 126: 53, 101: 53, 109: 52, 92: 52, 106: 50, 136: 49, 108: 49, 105: 49, 135: 48, 114: 48, 110: 47, 103: 47, 102: 47, 85: 47, 113: 46, 127: 45, 120: 45, 117: 45, 112: 45, 121: 44, 138: 42, 125: 41, 124: 41, 115: 41, 148: 40, 137: 40, 134: 40, 107: 40, 133: 39, 123: 39, 146: 37, 143: 37, 130: 37, 119: 36, 131: 35, 128: 35, 170: 33, 162: 33, 150: 33, 144: 33, 140: 33, 129: 33, 111: 33, 145: 32, 132: 32, 118: 32, 169: 31, 168: 30, 156: 30, 154: 30, 141: 30, 200: 29, 161: 29, 149: 29, 160: 28, 122: 28, 195: 27, 189: 27, 180: 27, 178: 27, 174: 27, 166: 27, 158: 27, 157: 27, 155: 27, 142: 27, 171: 26, 163: 26, 147: 26, 139: 26, 250: 25, 208: 25, 186: 25, 167: 25, 218: 24, 211: 24, 182: 24, 172: 24, 152: 24, 151: 24, 116: 24, 217: 23, 207: 23, 192: 23, 179: 23, 177: 23, 175: 23, 281: 22, 204: 22, 190: 22, 176: 22, 196: 21, 165: 21, 164: 21, 159: 21, 216: 20, 213: 20, 206: 20, 194: 20, 173: 20, 296: 19, 260: 19, 254: 19, 232: 19, 210: 19, 199: 19, 191: 19, 288: 18, 243: 18, 230: 18, 227: 18, 226: 18, 223: 18, 219: 18, 214: 18, 201: 18, 197: 18, 193: 18, 188: 18, 184: 18, 183: 18, 181: 18, 327: 17, 305: 17, 267: 17, 263: 17, 242: 17, 239: 17, 229: 17, 225: 17, 222: 17, 221: 17, 215: 17, 212: 17, 185: 17, 330: 16, 300: 16, 293: 16, 278: 16, 253: 16, 251: 16, 238: 16, 237: 16, 203: 16, 198: 16, 187: 16, 153: 16, 322: 15, 315: 15, 298: 15, 290: 15, 282: 15, 246: 15, 236: 15, 233: 15, 220: 15, 202: 15, 404: 14, 355: 14, 343: 14, 309: 14, 285: 14, 276: 14, 271: 14, 257: 14, 256: 14, 235: 14, 205: 14, 430: 13, 397: 13, 342: 13, 334: 13, 318: 13, 314: 13, 277: 13, 275: 13, 262: 13, 261: 13, 255: 13, 247: 13, 245: 13, 224: 13, 345: 12, 294: 12, 292: 12, 279: 12, 273: 12, 270: 12, 264: 12, 252: 12, 249: 12, 240: 12, 209: 12, 417: 11, 389: 11, 385: 11, 354: 11, 347: 11, 324: 11, 323: 11, 312: 11, 304: 11, 284: 11, 486: 10, 439: 10, 401: 10, 378: 10, 373: 10, 367: 10, 363: 10, 352: 10, 346: 10, 337: 10, 320: 10, 302: 10, 297: 10, 295: 10, 286: 10, 280: 10, 268: 10, 259: 10, 241: 10, 234: 10, 231: 10, 228: 10, 568: 9, 550: 9, 531: 9, 530: 9, 526: 9, 491: 9, 434: 9, 407: 9, 388: 9, 382: 9, 359: 9, 341: 9, 331: 9, 328: 9, 317: 9, 316: 9, 311: 9, 307: 9, 303: 9, 291: 9, 289: 9, 287: 9, 269: 9, 258: 9, 244: 9, 750: 8, 539: 8, 468: 8, 452: 8, 448: 8, 433: 8, 421: 8, 412: 8, 411: 8, 410: 8, 403: 8, 394: 8, 392: 8, 372: 8, 369: 8, 366: 8, 350: 8, 344: 8, 340: 8, 336: 8, 325: 8, 319: 8, 313: 8, 301: 8, 283: 8, 274: 8, 266: 8, 248: 8, 910: 7, 776: 7, 706: 7, 671: 7, 655: 7, 603: 7, 576: 7, 535: 7, 527: 7, 507: 7, 483: 7, 464: 7, 456: 7, 436: 7, 420: 7, 419: 7, 405: 7, 400: 7, 395: 7, 391: 7, 387: 7, 386: 7, 384: 7, 380: 7, 374: 7, 356: 7, 353: 7, 349: 7, 339: 7, 335: 7, 333: 7, 332: 7, 321: 7, 310: 7, 308: 7, 299: 7, 272: 7, 1123: 6, 873: 6, 800: 6, 785: 6, 775: 6, 752: 6, 698: 6, 696: 6, 685: 6, 672: 6, 661: 6, 657: 6, 652: 6, 651: 6, 644: 6, 618: 6, 609: 6, 597: 6, 593: 6, 585: 6, 557: 6, 544: 6, 540: 6, 532: 6, 524: 6, 522: 6, 510: 6, 506: 6, 505: 6, 504: 6, 496: 6, 489: 6, 487: 6, 484: 6, 477: 6, 470: 6, 467: 6, 449: 6, 409: 6, 408: 6, 399: 6, 376: 6, 375: 6, 370: 6, 365: 6, 361: 6, 360: 6, 348: 6, 338: 6, 329: 6, 1831: 5, 1427: 5, 1231: 5, 1129: 5, 1034: 5, 1032: 5, 953: 5, 935: 5, 927: 5, 850: 5, 805: 5, 781: 5, 764: 5, 746: 5, 745: 5, 743: 5, 729: 5, 718: 5, 695: 5, 694: 5, 677: 5, 673: 5, 665: 5, 662: 5, 659: 5, 649: 5, 648: 5, 647: 5, 637: 5, 634: 5, 632: 5, 617: 5, 616: 5, 613: 5, 607: 5, 594: 5, 578: 5, 573: 5, 570: 5, 561: 5, 538: 5, 537: 5, 529: 5, 513: 5, 512: 5, 511: 5, 500: 5, 490: 5, 488: 5, 485: 5, 471: 5, 469: 5, 459: 5, 457: 5, 446: 5, 442: 5, 441: 5, 440: 5, 438: 5, 432: 5, 431: 5, 428: 5, 427: 5, 426: 5, 424: 5, 415: 5, 414: 5, 390: 5, 383: 5, 358: 5, 326: 5, 3384: 4, 2054: 4, 1873: 4, 1863: 4, 1845: 4, 1601: 4, 1569: 4, 1567: 4, 1483: 4, 1362: 4, 1334: 4, 1326: 4, 1316: 4, 1301: 4, 1266: 4, 1203: 4, 1191: 4, 1157: 4, 1143: 4, 1137: 4, 1115: 4, 1106: 4, 1095: 4, 1015: 4, 993: 4, 976: 4, 973: 4, 966: 4, 952: 4, 945: 4, 938: 4, 930: 4, 898: 4, 893: 4, 891: 4, 883: 4, 882: 4, 875: 4, 871: 4, 859: 4, 856: 4, 852: 4, 837: 4, 836: 4, 832: 4, 801: 4, 795: 4, 791: 4, 773: 4, 772: 4, 771: 4, 767: 4, 762: 4, 755: 4, 751: 4, 749: 4, 748: 4, 742: 4, 739: 4, 734: 4, 731: 4, 728: 4, 725: 4, 721: 4, 700: 4, 681: 4, 668: 4, 658: 4, 642: 4, 635: 4, 633: 4, 629: 4, 628: 4, 627: 4, 625: 4, 623: 4, 614: 4, 612: 4, 611: 4, 600: 4, 598: 4, 596: 4, 591: 4, 588: 4, 584: 4, 582: 4, 579: 4, 574: 4, 571: 4, 566: 4, 559: 4, 554: 4, 553: 4, 549: 4, 543: 4, 542: 4, 541: 4, 533: 4, 515: 4, 514: 4, 509: 4, 497: 4, 494: 4, 492: 4, 482: 4, 481: 4, 479: 4, 478: 4, 473: 4, 466: 4, 463: 4, 461: 4, 460: 4, 458: 4, 455: 4, 454: 4, 453: 4, 447: 4, 445: 4, 437: 4, 435: 4, 429: 4, 416: 4, 413: 4, 406: 4, 402: 4, 398: 4, 393: 4, 371: 4, 368: 4, 364: 4, 362: 4, 351: 4, 306: 4, 4731: 3, 3249: 3, 3231: 3, 2608: 3, 2571: 3, 2437: 3, 2280: 3, 2231: 3, 2228: 3, 2220: 3, 2034: 3, 2030: 3, 2022: 3, 1788: 3, 1780: 3, 1684: 3, 1671: 3, 1644: 3, 1626: 3, 1563: 3, 1562: 3, 1550: 3, 1539: 3, 1511: 3, 1509: 3, 1508: 3, 1495: 3, 1491: 3, 1467: 3, 1392: 3, 1377: 3, 1355: 3, 1351: 3, 1320: 3, 1280: 3, 1274: 3, 1260: 3, 1259: 3, 1255: 3, 1244: 3, 1239: 3, 1230: 3, 1229: 3, 1216: 3, 1209: 3, 1186: 3, 1171: 3, 1168: 3, 1156: 3, 1153: 3, 1145: 3, 1142: 3, 1136: 3, 1082: 3, 1080: 3, 1071: 3, 1069: 3, 1062: 3, 1060: 3, 1053: 3, 1047: 3, 1041: 3, 1036: 3, 1024: 3, 1014: 3, 1013: 3, 1003: 3, 1002: 3, 992: 3, 990: 3, 977: 3, 972: 3, 970: 3, 965: 3, 957: 3, 954: 3, 951: 3, 950: 3, 948: 3, 947: 3, 942: 3, 932: 3, 917: 3, 911: 3, 908: 3, 906: 3, 895: 3, 870: 3, 868: 3, 866: 3, 865: 3, 864: 3, 855: 3, 849: 3, 845: 3, 843: 3, 842: 3, 840: 3, 838: 3, 818: 3, 814: 3, 812: 3, 811: 3, 807: 3, 802: 3, 796: 3, 794: 3, 792: 3, 784: 3, 777: 3, 774: 3, 766: 3, 758: 3, 754: 3, 753: 3, 747: 3, 738: 3, 724: 3, 719: 3, 713: 3, 710: 3, 709: 3, 703: 3, 697: 3, 693: 3, 691: 3, 688: 3, 686: 3, 679: 3, 676: 3, 670: 3, 666: 3, 664: 3, 656: 3, 646: 3, 640: 3, 639: 3, 638: 3, 636: 3, 626: 3, 620: 3, 619: 3, 615: 3, 610: 3, 608: 3, 604: 3, 590: 3, 589: 3, 587: 3, 575: 3, 572: 3, 569: 3, 564: 3, 560: 3, 548: 3, 547: 3, 545: 3, 523: 3, 520: 3, 519: 3, 517: 3, 508: 3, 502: 3, 501: 3, 493: 3, 480: 3, 476: 3, 475: 3, 474: 3, 462: 3, 451: 3, 450: 3, 443: 3, 423: 3, 422: 3, 379: 3, 377: 3, 8864: 2, 8658: 2, 7314: 2, 7088: 2, 6401: 2, 6172: 2, 6074: 2, 6071: 2, 5361: 2, 4482: 2, 4478: 2, 4255: 2, 4214: 2, 4202: 2, 4111: 2, 3965: 2, 3962: 2, 3750: 2, 3692: 2, 3681: 2, 3501: 2, 3467: 2, 3426: 2, 3420: 2, 3386: 2, 3383: 2, 3377: 2, 3361: 2, 3332: 2, 3315: 2, 3309: 2, 3278: 2, 3236: 2, 3147: 2, 3118: 2, 3111: 2, 3102: 2, 3051: 2, 3041: 2, 2982: 2, 2975: 2, 2904: 2, 2797: 2, 2757: 2, 2740: 2, 2700: 2, 2697: 2, 2676: 2, 2673: 2, 2650: 2, 2639: 2, 2634: 2, 2612: 2, 2605: 2, 2579: 2, 2578: 2, 2557: 2, 2502: 2, 2491: 2, 2479: 2, 2476: 2, 2469: 2, 2468: 2, 2456: 2, 2443: 2, 2403: 2, 2400: 2, 2382: 2, 2363: 2, 2354: 2, 2342: 2, 2327: 2, 2319: 2, 2309: 2, 2300: 2, 2268: 2, 2254: 2, 2240: 2, 2239: 2, 2234: 2, 2233: 2, 2217: 2, 2189: 2, 2170: 2, 2167: 2, 2142: 2, 2135: 2, 2121: 2, 2114: 2, 2102: 2, 2098: 2, 2086: 2, 2077: 2, 2060: 2, 2046: 2, 2044: 2, 2027: 2, 2006: 2, 1999: 2, 1982: 2, 1959: 2, 1951: 2, 1933: 2, 1929: 2, 1919: 2, 1910: 2, 1905: 2, 1898: 2, 1870: 2, 1862: 2, 1846: 2, 1840: 2, 1838: 2, 1836: 2, 1834: 2, 1827: 2, 1819: 2, 1815: 2, 1805: 2, 1801: 2, 1794: 2, 1792: 2, 1786: 2, 1773: 2, 1754: 2, 1729: 2, 1725: 2, 1723: 2, 1691: 2, 1679: 2, 1673: 2, 1662: 2, 1659: 2, 1653: 2, 1636: 2, 1634: 2, 1630: 2, 1624: 2, 1609: 2, 1607: 2, 1605: 2, 1598: 2, 1593: 2, 1592: 2, 1581: 2, 1580: 2, 1576: 2, 1572: 2, 1564: 2, 1559: 2, 1554: 2, 1552: 2, 1548: 2, 1547: 2, 1544: 2, 1533: 2, 1531: 2, 1522: 2, 1520: 2, 1517: 2, 1514: 2, 1505: 2, 1504: 2, 1487: 2, 1486: 2, 1472: 2, 1468: 2, 1463: 2, 1446: 2, 1442: 2, 1431: 2, 1425: 2, 1413: 2, 1410: 2, 1397: 2, 1391: 2, 1390: 2, 1389: 2, 1382: 2, 1380: 2, 1364: 2, 1361: 2, 1353: 2, 1347: 2, 1341: 2, 1340: 2, 1331: 2, 1329: 2, 1324: 2, 1319: 2, 1302: 2, 1300: 2, 1299: 2, 1293: 2, 1290: 2, 1289: 2, 1284: 2, 1278: 2, 1277: 2, 1273: 2, 1265: 2, 1261: 2, 1250: 2, 1249: 2, 1248: 2, 1243: 2, 1242: 2, 1241: 2, 1234: 2, 1228: 2, 1226: 2, 1225: 2, 1224: 2, 1218: 2, 1214: 2, 1213: 2, 1212: 2, 1211: 2, 1210: 2, 1204: 2, 1199: 2, 1198: 2, 1180: 2, 1174: 2, 1170: 2, 1166: 2, 1164: 2, 1161: 2, 1155: 2, 1150: 2, 1146: 2, 1140: 2, 1128: 2, 1121: 2, 1117: 2, 1116: 2, 1111: 2, 1109: 2, 1102: 2, 1101: 2, 1093: 2, 1087: 2, 1084: 2, 1077: 2, 1074: 2, 1067: 2, 1056: 2, 1055: 2, 1054: 2, 1050: 2, 1043: 2, 1042: 2, 1031: 2, 1027: 2, 1022: 2, 1020: 2, 1016: 2, 1010: 2, 1008: 2, 1005: 2, 1004: 2, 1001: 2, 998: 2, 997: 2, 987: 2, 985: 2, 984: 2, 982: 2, 971: 2, 969: 2, 968: 2, 958: 2, 956: 2, 943: 2, 941: 2, 937: 2, 925: 2, 920: 2, 913: 2, 909: 2, 905: 2, 903: 2, 900: 2, 897: 2, 896: 2, 888: 2, 887: 2, 886: 2, 884: 2, 881: 2, 880: 2, 878: 2, 877: 2, 874: 2, 869: 2, 858: 2, 857: 2, 854: 2, 851: 2, 848: 2, 846: 2, 844: 2, 834: 2, 828: 2, 825: 2, 809: 2, 806: 2, 803: 2, 797: 2, 787: 2, 782: 2, 780: 2, 779: 2, 778: 2, 770: 2, 769: 2, 768: 2, 765: 2, 761: 2, 760: 2, 759: 2, 756: 2, 741: 2, 733: 2, 732: 2, 723: 2, 720: 2, 716: 2, 715: 2, 714: 2, 712: 2, 711: 2, 707: 2, 705: 2, 702: 2, 687: 2, 684: 2, 683: 2, 682: 2, 680: 2, 660: 2, 641: 2, 624: 2, 621: 2, 605: 2, 601: 2, 595: 2, 592: 2, 586: 2, 583: 2, 581: 2, 565: 2, 563: 2, 558: 2, 556: 2, 536: 2, 528: 2, 525: 2, 518: 2, 499: 2, 498: 2, 495: 2, 472: 2, 465: 2, 444: 2, 425: 2, 418: 2, 381: 2, 357: 2, 265: 2, 149646: 1, 116901: 1, 80136: 1, 65997: 1, 65918: 1, 64851: 1, 64806: 1, 62779: 1, 62100: 1, 53690: 1, 52577: 1, 49913: 1, 48304: 1, 46912: 1, 46750: 1, 44747: 1, 43075: 1, 42164: 1, 41593: 1, 40520: 1, 40155: 1, 39823: 1, 39811: 1, 38645: 1, 38623: 1, 37112: 1, 36892: 1, 35530: 1, 35307: 1, 33560: 1, 33064: 1, 32896: 1, 32832: 1, 32597: 1, 32003: 1, 31794: 1, 30096: 1, 29447: 1, 27596: 1, 26384: 1, 26031: 1, 25988: 1, 25482: 1, 25473: 1, 24742: 1, 24588: 1, 24538: 1, 24507: 1, 24464: 1, 24162: 1, 24054: 1, 23482: 1, 23093: 1, 22921: 1, 22073: 1, 21903: 1, 21857: 1, 21483: 1, 21095: 1, 20790: 1, 20559: 1, 20182: 1, 19857: 1, 19552: 1, 19513: 1, 19428: 1, 18922: 1, 18849: 1, 18760: 1, 18708: 1, 18694: 1, 18594: 1, 18376: 1, 18228: 1, 18063: 1, 18031: 1, 17989: 1, 17953: 1, 17751: 1, 17635: 1, 17624: 1, 17569: 1, 17556: 1, 17466: 1, 17439: 1, 17171: 1, 17083: 1, 17055: 1, 16969: 1, 16836: 1, 16832: 1, 16636: 1, 16591: 1, 16523: 1, 16477: 1, 16449: 1, 16387: 1, 16210: 1, 16105: 1, 16056: 1, 15655: 1, 15555: 1, 15541: 1, 15333: 1, 15302: 1, 15288: 1, 15171: 1, 14995: 1, 14962: 1, 14949: 1, 14857: 1, 14845: 1, 14817: 1, 14549: 1, 14533: 1, 14277: 1, 14240: 1, 14060: 1, 14020: 1, 13728: 1, 13571: 1, 13558: 1, 13482: 1, 13334: 1, 13320: 1, 13236: 1, 13221: 1, 13179: 1, 13065: 1, 13045: 1, 13029: 1, 12829: 1, 12800: 1, 12761: 1, 12702: 1, 12696: 1, 12607: 1, 12519: 1, 12504: 1, 12500: 1, 12491: 1, 12460: 1, 12415: 1, 12396: 1, 12351: 1, 12304: 1, 12277: 1, 12206: 1, 12134: 1, 12119: 1, 12091: 1, 12078: 1, 12055: 1, 12045: 1, 12020: 1, 11969: 1, 11961: 1, 11951: 1, 11922: 1, 11908: 1, 11871: 1, 11827: 1, 11811: 1, 11805: 1, 11768: 1, 11755: 1, 11742: 1, 11672: 1, 11659: 1, 11617: 1, 11612: 1, 11579: 1, 11565: 1, 11523: 1, 11418: 1, 11413: 1, 11268: 1, 11169: 1, 11163: 1, 11050: 1, 10909: 1, 10808: 1, 10793: 1, 10735: 1, 10723: 1, 10591: 1, 10577: 1, 10562: 1, 10479: 1, 10462: 1, 10439: 1, 10327: 1, 10244: 1, 10196: 1, 10084: 1, 10039: 1, 9986: 1, 9976: 1, 9904: 1, 9889: 1, 9888: 1, 9886: 1, 9851: 1, 9838: 1, 9818: 1, 9797: 1, 9786: 1, 9733: 1, 9723: 1, 9714: 1, 9711: 1, 9664: 1, 9627: 1, 9616: 1, 9573: 1, 9565: 1, 9512: 1, 9499: 1, 9480: 1, 9365: 1, 9352: 1, 9269: 1, 9238: 1, 9224: 1, 9204: 1, 9158: 1, 9123: 1, 9013: 1, 8984: 1, 8981: 1, 8939: 1, 8921: 1, 8912: 1, 8910: 1, 8875: 1, 8865: 1, 8853: 1, 8761: 1, 8723: 1, 8702: 1, 8690: 1, 8641: 1, 8626: 1, 8605: 1, 8592: 1, 8514: 1, 8491: 1, 8453: 1, 8443: 1, 8422: 1, 8335: 1, 8332: 1, 8300: 1, 8281: 1, 8260: 1, 8229: 1, 8197: 1, 8189: 1, 8167: 1, 8133: 1, 8115: 1, 8063: 1, 8017: 1, 7992: 1, 7990: 1, 7986: 1, 7953: 1, 7944: 1, 7907: 1, 7902: 1, 7809: 1, 7795: 1, 7697: 1, 7657: 1, 7634: 1, 7629: 1, 7614: 1, 7608: 1, 7598: 1, 7583: 1, 7574: 1, 7533: 1, 7528: 1, 7525: 1, 7509: 1, 7501: 1, 7480: 1, 7405: 1, 7395: 1, 7377: 1, 7330: 1, 7321: 1, 7311: 1, 7290: 1, 7287: 1, 7274: 1, 7262: 1, 7253: 1, 7241: 1, 7233: 1, 7210: 1, 7199: 1, 7172: 1, 7171: 1, 7155: 1, 7151: 1, 7150: 1, 7145: 1, 7127: 1, 7126: 1, 7114: 1, 7103: 1, 7087: 1, 7063: 1, 7062: 1, 7042: 1, 7040: 1, 7039: 1, 7032: 1, 7029: 1, 7011: 1, 7008: 1, 6994: 1, 6993: 1, 6953: 1, 6937: 1, 6918: 1, 6905: 1, 6856: 1, 6853: 1, 6823: 1, 6772: 1, 6763: 1, 6752: 1, 6748: 1, 6715: 1, 6695: 1, 6688: 1, 6686: 1, 6682: 1, 6651: 1, 6645: 1, 6642: 1, 6594: 1, 6590: 1, 6582: 1, 6569: 1, 6556: 1, 6525: 1, 6522: 1, 6521: 1, 6508: 1, 6503: 1, 6487: 1, 6476: 1, 6460: 1, 6448: 1, 6400: 1, 6395: 1, 6386: 1, 6319: 1, 6309: 1, 6304: 1, 6298: 1, 6292: 1, 6269: 1, 6256: 1, 6222: 1, 6221: 1, 6167: 1, 6161: 1, 6138: 1, 6098: 1, 6086: 1, 6051: 1, 6043: 1, 6038: 1, 6021: 1, 6009: 1, 5999: 1, 5995: 1, 5994: 1, 5959: 1, 5951: 1, 5938: 1, 5924: 1, 5905: 1, 5896: 1, 5856: 1, 5844: 1, 5832: 1, 5827: 1, 5823: 1, 5812: 1, 5796: 1, 5788: 1, 5769: 1, 5764: 1, 5760: 1, 5739: 1, 5735: 1, 5728: 1, 5690: 1, 5665: 1, 5655: 1, 5642: 1, 5619: 1, 5590: 1, 5586: 1, 5566: 1, 5556: 1, 5549: 1, 5546: 1, 5520: 1, 5516: 1, 5510: 1, 5508: 1, 5507: 1, 5505: 1, 5503: 1, 5500: 1, 5483: 1, 5439: 1, 5417: 1, 5401: 1, 5393: 1, 5366: 1, 5358: 1, 5344: 1, 5335: 1, 5329: 1, 5326: 1, 5321: 1, 5296: 1, 5288: 1, 5276: 1, 5247: 1, 5234: 1, 5221: 1, 5219: 1, 5218: 1, 5205: 1, 5191: 1, 5185: 1, 5159: 1, 5146: 1, 5140: 1, 5118: 1, 5112: 1, 5106: 1, 5098: 1, 5096: 1, 5090: 1, 5068: 1, 5060: 1, 5047: 1, 5046: 1, 5034: 1, 5032: 1, 5025: 1, 5011: 1, 4998: 1, 4981: 1, 4975: 1, 4957: 1, 4932: 1, 4931: 1, 4928: 1, 4926: 1, 4910: 1, 4909: 1, 4908: 1, 4896: 1, 4890: 1, 4874: 1, 4873: 1, 4870: 1, 4869: 1, 4852: 1, 4843: 1, 4839: 1, 4828: 1, 4827: 1, 4814: 1, 4811: 1, 4792: 1, 4778: 1, 4777: 1, 4773: 1, 4763: 1, 4761: 1, 4759: 1, 4752: 1, 4737: 1, 4725: 1, 4724: 1, 4716: 1, 4701: 1, 4696: 1, 4682: 1, 4678: 1, 4673: 1, 4670: 1, 4657: 1, 4656: 1, 4642: 1, 4640: 1, 4633: 1, 4631: 1, 4630: 1, 4601: 1, 4574: 1, 4561: 1, 4533: 1, 4531: 1, 4526: 1, 4522: 1, 4512: 1, 4505: 1, 4502: 1, 4501: 1, 4490: 1, 4485: 1, 4471: 1, 4468: 1, 4465: 1, 4463: 1, 4461: 1, 4447: 1, 4445: 1, 4444: 1, 4431: 1, 4427: 1, 4425: 1, 4403: 1, 4401: 1, 4399: 1, 4396: 1, 4377: 1, 4364: 1, 4357: 1, 4351: 1, 4348: 1, 4319: 1, 4307: 1, 4291: 1, 4278: 1, 4269: 1, 4249: 1, 4243: 1, 4239: 1, 4222: 1, 4220: 1, 4218: 1, 4217: 1, 4195: 1, 4194: 1, 4193: 1, 4192: 1, 4191: 1, 4190: 1, 4181: 1, 4175: 1, 4170: 1, 4169: 1, 4163: 1, 4147: 1, 4145: 1, 4139: 1, 4117: 1, 4114: 1, 4112: 1, 4105: 1, 4102: 1, 4095: 1, 4083: 1, 4080: 1, 4065: 1, 4050: 1, 4048: 1, 4043: 1, 4040: 1, 4029: 1, 4025: 1, 4024: 1, 4023: 1, 4017: 1, 4015: 1, 4008: 1, 4002: 1, 4000: 1, 3993: 1, 3985: 1, 3984: 1, 3976: 1, 3975: 1, 3971: 1, 3968: 1, 3955: 1, 3949: 1, 3948: 1, 3947: 1, 3939: 1, 3927: 1, 3915: 1, 3898: 1, 3892: 1, 3886: 1, 3883: 1, 3880: 1, 3875: 1, 3874: 1, 3863: 1, 3862: 1, 3855: 1, 3854: 1, 3849: 1, 3848: 1, 3845: 1, 3840: 1, 3837: 1, 3835: 1, 3828: 1, 3819: 1, 3806: 1, 3761: 1, 3759: 1, 3752: 1, 3751: 1, 3746: 1, 3742: 1, 3740: 1, 3735: 1, 3720: 1, 3717: 1, 3715: 1, 3713: 1, 3712: 1, 3711: 1, 3708: 1, 3686: 1, 3676: 1, 3672: 1, 3670: 1, 3667: 1, 3664: 1, 3663: 1, 3660: 1, 3653: 1, 3648: 1, 3647: 1, 3643: 1, 3635: 1, 3631: 1, 3630: 1, 3628: 1, 3613: 1, 3612: 1, 3601: 1, 3594: 1, 3593: 1, 3586: 1, 3582: 1, 3568: 1, 3566: 1, 3554: 1, 3553: 1, 3549: 1, 3538: 1, 3533: 1, 3529: 1, 3528: 1, 3527: 1, 3522: 1, 3515: 1, 3498: 1, 3495: 1, 3491: 1, 3489: 1, 3488: 1, 3480: 1, 3475: 1, 3474: 1, 3470: 1, 3466: 1, 3464: 1, 3462: 1, 3461: 1, 3450: 1, 3447: 1, 3442: 1, 3431: 1, 3427: 1, 3414: 1, 3413: 1, 3412: 1, 3408: 1, 3406: 1, 3400: 1, 3397: 1, 3395: 1, 3390: 1, 3379: 1, 3374: 1, 3365: 1, 3360: 1, 3357: 1, 3354: 1, 3348: 1, 3346: 1, 3338: 1, 3336: 1, 3331: 1, 3325: 1, 3318: 1, 3316: 1, 3313: 1, 3312: 1, 3304: 1, 3297: 1, 3289: 1, 3282: 1, 3271: 1, 3267: 1, 3266: 1, 3265: 1, 3262: 1, 3257: 1, 3253: 1, 3242: 1, 3230: 1, 3222: 1, 3217: 1, 3216: 1, 3211: 1, 3209: 1, 3199: 1, 3198: 1, 3193: 1, 3190: 1, 3185: 1, 3184: 1, 3182: 1, 3181: 1, 3180: 1, 3179: 1, 3164: 1, 3159: 1, 3154: 1, 3151: 1, 3139: 1, 3132: 1, 3120: 1, 3101: 1, 3100: 1, 3098: 1, 3092: 1, 3088: 1, 3075: 1, 3068: 1, 3061: 1, 3056: 1, 3042: 1, 3032: 1, 3026: 1, 3025: 1, 3018: 1, 3015: 1, 3014: 1, 3013: 1, 3012: 1, 3003: 1, 3002: 1, 2999: 1, 2998: 1, 2996: 1, 2993: 1, 2989: 1, 2983: 1, 2981: 1, 2974: 1, 2968: 1, 2967: 1, 2965: 1, 2960: 1, 2951: 1, 2939: 1, 2938: 1, 2935: 1, 2932: 1, 2926: 1, 2916: 1, 2914: 1, 2909: 1, 2900: 1, 2898: 1, 2896: 1, 2895: 1, 2893: 1, 2879: 1, 2878: 1, 2875: 1, 2873: 1, 2866: 1, 2863: 1, 2859: 1, 2858: 1, 2856: 1, 2844: 1, 2842: 1, 2838: 1, 2836: 1, 2823: 1, 2819: 1, 2817: 1, 2815: 1, 2812: 1, 2811: 1, 2810: 1, 2802: 1, 2800: 1, 2798: 1, 2787: 1, 2776: 1, 2770: 1, 2755: 1, 2754: 1, 2752: 1, 2751: 1, 2745: 1, 2738: 1, 2723: 1, 2722: 1, 2705: 1, 2704: 1, 2703: 1, 2695: 1, 2688: 1, 2683: 1, 2682: 1, 2680: 1, 2663: 1, 2660: 1, 2651: 1, 2647: 1, 2643: 1, 2635: 1, 2630: 1, 2629: 1, 2628: 1, 2626: 1, 2624: 1, 2623: 1, 2622: 1, 2615: 1, 2614: 1, 2607: 1, 2606: 1, 2604: 1, 2603: 1, 2601: 1, 2594: 1, 2591: 1, 2589: 1, 2588: 1, 2581: 1, 2580: 1, 2563: 1, 2553: 1, 2550: 1, 2547: 1, 2546: 1, 2541: 1, 2535: 1, 2534: 1, 2532: 1, 2528: 1, 2525: 1, 2523: 1, 2521: 1, 2520: 1, 2506: 1, 2503: 1, 2500: 1, 2498: 1, 2490: 1, 2484: 1, 2478: 1, 2477: 1, 2472: 1, 2453: 1, 2452: 1, 2451: 1, 2450: 1, 2448: 1, 2444: 1, 2436: 1, 2434: 1, 2427: 1, 2425: 1, 2419: 1, 2418: 1, 2416: 1, 2415: 1, 2414: 1, 2409: 1, 2407: 1, 2406: 1, 2402: 1, 2401: 1, 2399: 1, 2394: 1, 2391: 1, 2384: 1, 2383: 1, 2374: 1, 2373: 1, 2368: 1, 2367: 1, 2365: 1, 2360: 1, 2356: 1, 2347: 1, 2345: 1, 2343: 1, 2341: 1, 2338: 1, 2335: 1, 2334: 1, 2330: 1, 2329: 1, 2325: 1, 2322: 1, 2321: 1, 2316: 1, 2315: 1, 2314: 1, 2307: 1, 2305: 1, 2298: 1, 2294: 1, 2293: 1, 2290: 1, 2289: 1, 2287: 1, 2286: 1, 2284: 1, 2278: 1, 2277: 1, 2276: 1, 2273: 1, 2263: 1, 2252: 1, 2245: 1, 2242: 1, 2241: 1, 2238: 1, 2235: 1, 2230: 1, 2225: 1, 2224: 1, 2223: 1, 2222: 1, 2219: 1, 2214: 1, 2213: 1, 2211: 1, 2209: 1, 2207: 1, 2206: 1, 2200: 1, 2195: 1, 2194: 1, 2193: 1, 2191: 1, 2190: 1, 2188: 1, 2183: 1, 2181: 1, 2175: 1, 2169: 1, 2168: 1, 2166: 1, 2160: 1, 2158: 1, 2155: 1, 2154: 1, 2152: 1, 2144: 1, 2140: 1, 2139: 1, 2132: 1, 2125: 1, 2120: 1, 2115: 1, 2112: 1, 2110: 1, 2108: 1, 2107: 1, 2106: 1, 2099: 1, 2095: 1, 2090: 1, 2083: 1, 2080: 1, 2079: 1, 2076: 1, 2074: 1, 2072: 1, 2068: 1, 2067: 1, 2066: 1, 2065: 1, 2064: 1, 2061: 1, 2056: 1, 2048: 1, 2047: 1, 2045: 1, 2038: 1, 2037: 1, 2029: 1, 2026: 1, 2023: 1, 2020: 1, 2019: 1, 2018: 1, 2016: 1, 2015: 1, 2000: 1, 1998: 1, 1988: 1, 1987: 1, 1985: 1, 1983: 1, 1979: 1, 1972: 1, 1967: 1, 1964: 1, 1963: 1, 1961: 1, 1960: 1, 1957: 1, 1954: 1, 1952: 1, 1950: 1, 1949: 1, 1948: 1, 1939: 1, 1935: 1, 1931: 1, 1930: 1, 1926: 1, 1925: 1, 1924: 1, 1923: 1, 1916: 1, 1912: 1, 1908: 1, 1906: 1, 1904: 1, 1903: 1, 1900: 1, 1896: 1, 1895: 1, 1891: 1, 1888: 1, 1886: 1, 1884: 1, 1883: 1, 1882: 1, 1879: 1, 1877: 1, 1875: 1, 1874: 1, 1872: 1, 1871: 1, 1869: 1, 1868: 1, 1866: 1, 1864: 1, 1861: 1, 1860: 1, 1859: 1, 1855: 1, 1851: 1, 1843: 1, 1842: 1, 1837: 1, 1833: 1, 1826: 1, 1825: 1, 1813: 1, 1811: 1, 1807: 1, 1802: 1, 1797: 1, 1793: 1, 1789: 1, 1785: 1, 1783: 1, 1782: 1, 1779: 1, 1778: 1, 1774: 1, 1767: 1, 1766: 1, 1764: 1, 1763: 1, 1761: 1, 1759: 1, 1755: 1, 1753: 1, 1752: 1, 1750: 1, 1749: 1, 1747: 1, 1745: 1, 1743: 1, 1740: 1, 1733: 1, 1732: 1, 1728: 1, 1721: 1, 1720: 1, 1716: 1, 1715: 1, 1714: 1, 1713: 1, 1711: 1, 1710: 1, 1707: 1, 1706: 1, 1705: 1, 1702: 1, 1699: 1, 1698: 1, 1697: 1, 1694: 1, 1693: 1, 1692: 1, 1688: 1, 1687: 1, 1686: 1, 1685: 1, 1670: 1, 1664: 1, 1661: 1, 1657: 1, 1656: 1, 1655: 1, 1654: 1, 1651: 1, 1648: 1, 1645: 1, 1643: 1, 1642: 1, 1641: 1, 1639: 1, 1637: 1, 1635: 1, 1632: 1, 1628: 1, 1625: 1, 1619: 1, 1618: 1, 1616: 1, 1612: 1, 1611: 1, 1610: 1, 1603: 1, 1600: 1, 1596: 1, 1595: 1, 1594: 1, 1591: 1, 1590: 1, 1585: 1, 1584: 1, 1583: 1, 1578: 1, 1577: 1, 1575: 1, 1570: 1, 1566: 1, 1565: 1, 1557: 1, 1553: 1, 1549: 1, 1542: 1, 1540: 1, 1536: 1, 1535: 1, 1530: 1, 1529: 1, 1527: 1, 1525: 1, 1524: 1, 1523: 1, 1521: 1, 1512: 1, 1510: 1, 1507: 1, 1506: 1, 1503: 1, 1502: 1, 1494: 1, 1492: 1, 1488: 1, 1481: 1, 1478: 1, 1477: 1, 1476: 1, 1475: 1, 1474: 1, 1473: 1, 1470: 1, 1469: 1, 1466: 1, 1465: 1, 1462: 1, 1460: 1, 1458: 1, 1457: 1, 1454: 1, 1453: 1, 1452: 1, 1451: 1, 1450: 1, 1448: 1, 1445: 1, 1437: 1, 1434: 1, 1433: 1, 1432: 1, 1430: 1, 1429: 1, 1424: 1, 1423: 1, 1422: 1, 1421: 1, 1418: 1, 1415: 1, 1414: 1, 1412: 1, 1411: 1, 1400: 1, 1395: 1, 1393: 1, 1388: 1, 1387: 1, 1386: 1, 1384: 1, 1381: 1, 1379: 1, 1378: 1, 1376: 1, 1375: 1, 1373: 1, 1372: 1, 1371: 1, 1370: 1, 1368: 1, 1367: 1, 1366: 1, 1357: 1, 1356: 1, 1346: 1, 1343: 1, 1338: 1, 1337: 1, 1335: 1, 1332: 1, 1330: 1, 1327: 1, 1325: 1, 1323: 1, 1314: 1, 1312: 1, 1311: 1, 1310: 1, 1309: 1, 1306: 1, 1303: 1, 1296: 1, 1295: 1, 1294: 1, 1292: 1, 1288: 1, 1286: 1, 1285: 1, 1282: 1, 1281: 1, 1279: 1, 1276: 1, 1272: 1, 1271: 1, 1268: 1, 1267: 1, 1264: 1, 1262: 1, 1258: 1, 1254: 1, 1253: 1, 1251: 1, 1247: 1, 1245: 1, 1240: 1, 1238: 1, 1237: 1, 1236: 1, 1235: 1, 1227: 1, 1223: 1, 1221: 1, 1217: 1, 1215: 1, 1208: 1, 1206: 1, 1205: 1, 1201: 1, 1200: 1, 1197: 1, 1196: 1, 1195: 1, 1194: 1, 1193: 1, 1189: 1, 1188: 1, 1187: 1, 1185: 1, 1183: 1, 1182: 1, 1179: 1, 1176: 1, 1175: 1, 1173: 1, 1172: 1, 1169: 1, 1165: 1, 1160: 1, 1159: 1, 1154: 1, 1151: 1, 1149: 1, 1147: 1, 1141: 1, 1135: 1, 1132: 1, 1131: 1, 1130: 1, 1127: 1, 1125: 1, 1124: 1, 1120: 1, 1118: 1, 1113: 1, 1112: 1, 1108: 1, 1107: 1, 1104: 1, 1103: 1, 1100: 1, 1098: 1, 1097: 1, 1094: 1, 1092: 1, 1091: 1, 1089: 1, 1088: 1, 1086: 1, 1085: 1, 1079: 1, 1076: 1, 1073: 1, 1072: 1, 1068: 1, 1064: 1, 1063: 1, 1061: 1, 1059: 1, 1058: 1, 1049: 1, 1048: 1, 1046: 1, 1044: 1, 1040: 1, 1039: 1, 1038: 1, 1033: 1, 1030: 1, 1029: 1, 1028: 1, 1023: 1, 1021: 1, 1017: 1, 1011: 1, 1009: 1, 995: 1, 994: 1, 991: 1, 989: 1, 988: 1, 983: 1, 979: 1, 978: 1, 975: 1, 967: 1, 963: 1, 961: 1, 960: 1, 955: 1, 949: 1, 946: 1, 944: 1, 939: 1, 936: 1, 929: 1, 928: 1, 926: 1, 922: 1, 921: 1, 919: 1, 918: 1, 902: 1, 901: 1, 899: 1, 894: 1, 890: 1, 889: 1, 885: 1, 879: 1, 876: 1, 872: 1, 867: 1, 862: 1, 861: 1, 853: 1, 847: 1, 841: 1, 831: 1, 829: 1, 827: 1, 826: 1, 824: 1, 823: 1, 822: 1, 819: 1, 815: 1, 810: 1, 804: 1, 799: 1, 793: 1, 789: 1, 788: 1, 786: 1, 763: 1, 757: 1, 744: 1, 736: 1, 735: 1, 730: 1, 727: 1, 726: 1, 722: 1, 717: 1, 708: 1, 704: 1, 699: 1, 692: 1, 689: 1, 678: 1, 674: 1, 669: 1, 663: 1, 654: 1, 650: 1, 643: 1, 630: 1, 622: 1, 606: 1, 602: 1, 599: 1, 580: 1, 577: 1, 562: 1, 546: 1, 534: 1, 521: 1, 516: 1, 503: 1, 396: 1})



```python
cv_log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_text_feature_onehotCoding, y_train)
    
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_text_feature_onehotCoding, y_train)
    predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)
    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
```

    For values of alpha =  1e-05 The log loss is: 1.3657925584554536
    For values of alpha =  0.0001 The log loss is: 1.3230936320175102
    For values of alpha =  0.001 The log loss is: 1.161459815815937
    For values of alpha =  0.01 The log loss is: 1.226698390747041
    For values of alpha =  0.1 The log loss is: 1.445551000886829
    For values of alpha =  1 The log loss is: 1.6674060577089198



```python
fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_93_0.png)



```python
best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_text_feature_onehotCoding, y_train)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_text_feature_onehotCoding, y_train)

predict_y = sig_clf.predict_proba(train_text_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_text_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

```

    For values of best alpha =  0.001 The train log loss is: 0.7631944987681992
    For values of best alpha =  0.001 The cross validation log loss is: 1.161459815815937
    For values of best alpha =  0.001 The test log loss is: 1.182219119184948



```python
def get_intersec_text(df):
    df_text_vec = CountVectorizer(min_df=3)
    df_text_fea = df_text_vec.fit_transform(df['TEXT'])
    df_text_features = df_text_vec.get_feature_names()

    df_text_fea_counts = df_text_fea.sum(axis=0).A1
    df_text_fea_dict = dict(zip(list(df_text_features),df_text_fea_counts))
    len1 = len(set(df_text_features))
    len2 = len(set(train_text_features) & set(df_text_features))
    return len1,len2
```


```python
len1,len2 = get_intersec_text(test_df)
print(np.round((len2/len1)*100, 3), "% of word of test data appeared in train data")
len1,len2 = get_intersec_text(cv_df)
print(np.round((len2/len1)*100, 3), "% of word of Cross Validation appeared in train data")
```

    95.835 % of word of test data appeared in train data
    96.185 % of word of Cross Validation appeared in train data


Three variables are all important because their log loss all smaller than random.

## Data prepration for Machine Learning models


```python
# Some useful functions we are going to use.
def report_log_loss(train_x, train_y, test_x, test_y,  clf):
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x, train_y)
    sig_clf_probs = sig_clf.predict_proba(test_x)
    return log_loss(test_y, sig_clf_probs, eps=1e-15)

def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    
    A =(((C.T)/(C.sum(axis=1))).T)
    
    B =(C/C.sum(axis=0)) 
    labels = [1,2,3,4,5,6,7,8,9]
    # representing A in heatmap format
    print("-"*20, "Confusion matrix", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    
    # representing B in heatmap format
    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()


def predict_and_plot_confusion_matrix(train_x, train_y,test_x, test_y, clf):
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x, train_y)
    pred_y = sig_clf.predict(test_x)

    # for calculating log_loss we willl provide the array of probabilities belongs to each class
    print("Log loss :",log_loss(test_y, sig_clf.predict_proba(test_x)))
    # calculating the number of data points that are misclassified
    print("Number of mis-classified points :", np.count_nonzero((pred_y- test_y))/test_y.shape[0])
    plot_confusion_matrix(test_y, pred_y)
    
    
# this function will be used just for naive bayes
# for the given indices, we will print the name of the features
# and we will check whether the feature present in the test point text or not
def get_impfeature_names(indices, text, gene, var, no_features):
    gene_count_vec = CountVectorizer()
    var_count_vec = CountVectorizer()
    text_count_vec = CountVectorizer(min_df=3)
    
    gene_vec = gene_count_vec.fit(train_df['Gene'])
    var_vec  = var_count_vec.fit(train_df['Variation'])
    text_vec = text_count_vec.fit(train_df['TEXT'])
    
    fea1_len = len(gene_vec.get_feature_names())
    fea2_len = len(var_count_vec.get_feature_names())
    
    word_present = 0
    for i,v in enumerate(indices):
        if (v < fea1_len):
            word = gene_vec.get_feature_names()[v]
            yes_no = True if word == gene else False
            if yes_no:
                word_present += 1
                print(i, "Gene feature [{}] present in test data point [{}]".format(word,yes_no))
        elif (v < fea1_len+fea2_len):
            word = var_vec.get_feature_names()[v-(fea1_len)]
            yes_no = True if word == var else False
            if yes_no:
                word_present += 1
                print(i, "variation feature [{}] present in test data point [{}]".format(word,yes_no))
        else:
            word = text_vec.get_feature_names()[v-(fea1_len+fea2_len)]
            yes_no = True if word in text.split() else False
            if yes_no:
                word_present += 1
                print(i, "Text feature [{}] present in test data point [{}]".format(word,yes_no))

    print("Out of the top ",no_features," features ", word_present, "are present in query point")   
```

### Combine Features


```python
train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))
test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))
cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))

train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()
train_y = np.array(list(train_df['Class']))

test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()
test_y = np.array(list(test_df['Class']))

cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()
cv_y = np.array(list(cv_df['Class']))


train_gene_var_responseCoding = np.hstack((train_gene_feature_responseCoding,train_variation_feature_responseCoding))
test_gene_var_responseCoding = np.hstack((test_gene_feature_responseCoding,test_variation_feature_responseCoding))
cv_gene_var_responseCoding = np.hstack((cv_gene_feature_responseCoding,cv_variation_feature_responseCoding))

train_x_responseCoding = np.hstack((train_gene_var_responseCoding, train_text_feature_responseCoding))
test_x_responseCoding = np.hstack((test_gene_var_responseCoding, test_text_feature_responseCoding))
cv_x_responseCoding = np.hstack((cv_gene_var_responseCoding, cv_text_feature_responseCoding))

```


```python
print("One hot encoding features :")
print("(number of data points * number of features) in train data = ", train_x_onehotCoding.shape)
print("(number of data points * number of features) in test data = ", test_x_onehotCoding.shape)
print("(number of data points * number of features) in cross validation data =", cv_x_onehotCoding.shape)
```

    One hot encoding features :
    (number of data points * number of features) in train data =  (2124, 54775)
    (number of data points * number of features) in test data =  (665, 54775)
    (number of data points * number of features) in cross validation data = (532, 54775)



```python
print(" Response encoding features :")
print("(number of data points * number of features) in train data = ", train_x_responseCoding.shape)
print("(number of data points * number of features) in test data = ", test_x_responseCoding.shape)
print("(number of data points * number of features) in cross validation data =", cv_x_responseCoding.shape)
```

     Response encoding features :
    (number of data points * number of features) in train data =  (2124, 27)
    (number of data points * number of features) in test data =  (665, 27)
    (number of data points * number of features) in cross validation data = (532, 27)


# Naive Bayes


```python
alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]
cv_log_error_array = []
for i in alpha:
    print("for alpha =", i)
    clf = MultinomialNB(alpha=i)
    clf.fit(train_x_onehotCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_onehotCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    # to avoid rounding error while multiplying probabilites we use log-probability estimates
    print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 
```

    for alpha = 1e-05
    Log Loss : 1.2679073986880116
    for alpha = 0.0001
    Log Loss : 1.253207047668102
    for alpha = 0.001
    Log Loss : 1.2525019714213907
    for alpha = 0.1
    Log Loss : 1.259890417929449
    for alpha = 1
    Log Loss : 1.2445351057845766
    for alpha = 10
    Log Loss : 1.3660119289785717
    for alpha = 100
    Log Loss : 1.3721137429591654
    for alpha = 1000
    Log Loss : 1.3295731374980182



```python
fig, ax = plt.subplots()
ax.plot(np.log10(alpha), cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (np.log10(alpha[i]),cv_log_error_array[i]))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_106_0.png)



```python
best_alpha = np.argmin(cv_log_error_array)
clf = MultinomialNB(alpha=alpha[best_alpha])
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)


predict_y = sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

```

    For values of best alpha =  1 The train log loss is: 0.9182035794671206
    For values of best alpha =  1 The cross validation log loss is: 1.2445351057845766
    For values of best alpha =  1 The test log loss is: 1.2871686792295556



```python
clf = MultinomialNB(alpha=alpha[best_alpha])
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)
sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
# to avoid rounding error while multiplying probabilites we use log-probability estimates
print("Log Loss :",log_loss(cv_y, sig_clf_probs))
print("Number of missclassified point :", np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- cv_y))/cv_y.shape[0])
plot_confusion_matrix(cv_y, sig_clf.predict(cv_x_onehotCoding.toarray()))
```

    Log Loss : 1.2445351057845766
    Number of missclassified point : 0.37218045112781956
    -------------------- Confusion matrix --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_108_1.png)


    -------------------- Precision matrix (Columm Sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_108_3.png)


    -------------------- Recall matrix (Row sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_108_5.png)



```python
# try to interpret this model, we look at two points.
test_point_index = 3
no_feature = 100
predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :", predicted_cls[0])
print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))
print("Actual Class :", test_y[test_point_index])
indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]
print("-"*50)
get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
```

    Predicted Class : 7
    Predicted Class Probabilities: [[0.0696 0.0862 0.0213 0.0939 0.0506 0.0374 0.6317 0.0047 0.0045]]
    Actual Class : 7
    --------------------------------------------------
    17 Text feature [presence] present in test data point [True]
    18 Text feature [kinase] present in test data point [True]
    19 Text feature [downstream] present in test data point [True]
    21 Text feature [inhibitor] present in test data point [True]
    22 Text feature [activating] present in test data point [True]
    23 Text feature [independent] present in test data point [True]
    25 Text feature [potential] present in test data point [True]
    26 Text feature [recently] present in test data point [True]
    27 Text feature [activation] present in test data point [True]
    28 Text feature [well] present in test data point [True]
    29 Text feature [cell] present in test data point [True]
    30 Text feature [also] present in test data point [True]
    31 Text feature [cells] present in test data point [True]
    32 Text feature [mutations] present in test data point [True]
    33 Text feature [shown] present in test data point [True]
    34 Text feature [previously] present in test data point [True]
    35 Text feature [however] present in test data point [True]
    36 Text feature [similar] present in test data point [True]
    37 Text feature [higher] present in test data point [True]
    38 Text feature [10] present in test data point [True]
    39 Text feature [addition] present in test data point [True]
    40 Text feature [showed] present in test data point [True]
    42 Text feature [growth] present in test data point [True]
    43 Text feature [treated] present in test data point [True]
    44 Text feature [observed] present in test data point [True]
    46 Text feature [found] present in test data point [True]
    47 Text feature [obtained] present in test data point [True]
    48 Text feature [factor] present in test data point [True]
    49 Text feature [total] present in test data point [True]
    51 Text feature [may] present in test data point [True]
    54 Text feature [inhibitors] present in test data point [True]
    55 Text feature [without] present in test data point [True]
    57 Text feature [described] present in test data point [True]
    58 Text feature [studies] present in test data point [True]
    59 Text feature [respectively] present in test data point [True]
    60 Text feature [figure] present in test data point [True]
    61 Text feature [12] present in test data point [True]
    62 Text feature [concentrations] present in test data point [True]
    63 Text feature [confirmed] present in test data point [True]
    64 Text feature [mutation] present in test data point [True]
    65 Text feature [various] present in test data point [True]
    66 Text feature [proliferation] present in test data point [True]
    68 Text feature [suggests] present in test data point [True]
    69 Text feature [followed] present in test data point [True]
    71 Text feature [could] present in test data point [True]
    73 Text feature [report] present in test data point [True]
    74 Text feature [sensitive] present in test data point [True]
    76 Text feature [inhibited] present in test data point [True]
    77 Text feature [thus] present in test data point [True]
    78 Text feature [reported] present in test data point [True]
    79 Text feature [consistent] present in test data point [True]
    80 Text feature [identified] present in test data point [True]
    81 Text feature [approximately] present in test data point [True]
    82 Text feature [including] present in test data point [True]
    83 Text feature [3a] present in test data point [True]
    84 Text feature [new] present in test data point [True]
    85 Text feature [phosphorylation] present in test data point [True]
    86 Text feature [using] present in test data point [True]
    88 Text feature [although] present in test data point [True]
    89 Text feature [different] present in test data point [True]
    90 Text feature [demonstrated] present in test data point [True]
    92 Text feature [due] present in test data point [True]
    93 Text feature [three] present in test data point [True]
    95 Text feature [3b] present in test data point [True]
    96 Text feature [approved] present in test data point [True]
    97 Text feature [hours] present in test data point [True]
    98 Text feature [small] present in test data point [True]
    Out of the top  100  features  67 are present in query point



```python
test_point_index = 100
no_feature = 100
predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :", predicted_cls[0])
print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))
print("Actual Class :", test_y[test_point_index])
indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]
print("-"*50)
get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
```

    Predicted Class : 4
    Predicted Class Probabilities: [[0.0654 0.0798 0.02   0.6891 0.0475 0.035  0.0545 0.0044 0.0042]]
    Actual Class : 4
    --------------------------------------------------
    10 Text feature [function] present in test data point [True]
    11 Text feature [protein] present in test data point [True]
    14 Text feature [missense] present in test data point [True]
    15 Text feature [proteins] present in test data point [True]
    16 Text feature [activity] present in test data point [True]
    17 Text feature [acid] present in test data point [True]
    18 Text feature [retained] present in test data point [True]
    19 Text feature [amino] present in test data point [True]
    20 Text feature [mammalian] present in test data point [True]
    21 Text feature [experiments] present in test data point [True]
    23 Text feature [functional] present in test data point [True]
    24 Text feature [suppressor] present in test data point [True]
    25 Text feature [results] present in test data point [True]
    28 Text feature [critical] present in test data point [True]
    29 Text feature [determined] present in test data point [True]
    31 Text feature [purified] present in test data point [True]
    33 Text feature [type] present in test data point [True]
    34 Text feature [partially] present in test data point [True]
    35 Text feature [whereas] present in test data point [True]
    36 Text feature [terminal] present in test data point [True]
    39 Text feature [stability] present in test data point [True]
    40 Text feature [buffer] present in test data point [True]
    41 Text feature [ability] present in test data point [True]
    42 Text feature [containing] present in test data point [True]
    43 Text feature [acids] present in test data point [True]
    44 Text feature [wild] present in test data point [True]
    45 Text feature [affect] present in test data point [True]
    46 Text feature [indicate] present in test data point [True]
    47 Text feature [bind] present in test data point [True]
    48 Text feature [whether] present in test data point [True]
    49 Text feature [sds] present in test data point [True]
    50 Text feature [loss] present in test data point [True]
    52 Text feature [transfection] present in test data point [True]
    53 Text feature [indicates] present in test data point [True]
    54 Text feature [made] present in test data point [True]
    60 Text feature [transfected] present in test data point [True]
    62 Text feature [determine] present in test data point [True]
    63 Text feature [assay] present in test data point [True]
    66 Text feature [vector] present in test data point [True]
    67 Text feature [mm] present in test data point [True]
    68 Text feature [washed] present in test data point [True]
    69 Text feature [therefore] present in test data point [True]
    70 Text feature [indicated] present in test data point [True]
    71 Text feature [shown] present in test data point [True]
    72 Text feature [important] present in test data point [True]
    73 Text feature [although] present in test data point [True]
    75 Text feature [either] present in test data point [True]
    77 Text feature [tested] present in test data point [True]
    79 Text feature [related] present in test data point [True]
    80 Text feature [generated] present in test data point [True]
    81 Text feature [correspond] present in test data point [True]
    82 Text feature [system] present in test data point [True]
    84 Text feature [binding] present in test data point [True]
    85 Text feature [cannot] present in test data point [True]
    87 Text feature [predicted] present in test data point [True]
    88 Text feature [vitro] present in test data point [True]
    90 Text feature [possible] present in test data point [True]
    91 Text feature [tris] present in test data point [True]
    92 Text feature [germline] present in test data point [True]
    93 Text feature [escherichia] present in test data point [True]
    96 Text feature [associated] present in test data point [True]
    97 Text feature [see] present in test data point [True]
    98 Text feature [analyzed] present in test data point [True]
    99 Text feature [two] present in test data point [True]
    Out of the top  100  features  64 are present in query point


# KNN


```python
alpha = [5, 11, 15, 21, 31, 41, 51, 99]
cv_log_error_array = []
for i in alpha:
    print("for alpha =", i)
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(train_x_responseCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_responseCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    # to avoid rounding error while multiplying probabilites we use log-probability estimates
    print("Log Loss :",log_loss(cv_y, sig_clf_probs))
```

    for alpha = 5
    Log Loss : 1.0686700404111178
    for alpha = 11
    Log Loss : 1.0489573652859332
    for alpha = 15
    Log Loss : 1.036299778737663
    for alpha = 21
    Log Loss : 1.054752223903923
    for alpha = 31
    Log Loss : 1.054471066705125
    for alpha = 41
    Log Loss : 1.0590535944663126
    for alpha = 51
    Log Loss : 1.062149431204017
    for alpha = 99
    Log Loss : 1.0814822646995925



```python
fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_113_0.png)



```python
best_alpha = np.argmin(cv_log_error_array)
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])
clf.fit(train_x_responseCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_responseCoding, train_y)

predict_y = sig_clf.predict_proba(train_x_responseCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_x_responseCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_x_responseCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

```

    For values of best alpha =  15 The train log loss is: 0.7144516872224927
    For values of best alpha =  15 The cross validation log loss is: 1.036299778737663
    For values of best alpha =  15 The test log loss is: 1.0896767578236566



```python
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])
predict_and_plot_confusion_matrix(train_x_responseCoding, train_y, cv_x_responseCoding, cv_y, clf)
```

    Log loss : 1.036299778737663
    Number of mis-classified points : 0.39849624060150374
    -------------------- Confusion matrix --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_115_1.png)


    -------------------- Precision matrix (Columm Sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_115_3.png)


    -------------------- Recall matrix (Row sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_115_5.png)



```python
# Lets look at few test points
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])
clf.fit(train_x_responseCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_responseCoding, train_y)

test_point_index = 1
predicted_cls = sig_clf.predict(test_x_responseCoding[0].reshape(1,-1))
print("Predicted Class :", predicted_cls[0])
print("Actual Class :", test_y[test_point_index])
neighbors = clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1, -1), alpha[best_alpha])
print("The ",alpha[best_alpha]," nearest neighbours of the test points belongs to classes",train_y[neighbors[1][0]])
print("Fequency of nearest points :",Counter(train_y[neighbors[1][0]]))
```

    Predicted Class : 7
    Actual Class : 7
    The  15  nearest neighbours of the test points belongs to classes [7 7 7 7 2 7 2 6 7 7 7 6 7 7 2]
    Fequency of nearest points : Counter({7: 10, 2: 3, 6: 2})



```python
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])
clf.fit(train_x_responseCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_responseCoding, train_y)

test_point_index = 100

predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))
print("Predicted Class :", predicted_cls[0])
print("Actual Class :", test_y[test_point_index])
neighbors = clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1, -1), alpha[best_alpha])
print("the k value for knn is",alpha[best_alpha],"and the nearest neighbours of the test points belongs to classes",train_y[neighbors[1][0]])
print("Fequency of nearest points :",Counter(train_y[neighbors[1][0]]))
```

    Predicted Class : 4
    Actual Class : 4
    the k value for knn is 15 and the nearest neighbours of the test points belongs to classes [4 4 4 4 4 4 1 4 4 4 1 4 4 4 4]
    Fequency of nearest points : Counter({4: 13, 1: 2})


**We know there are 9 classes of cancer though**


```python
clf = KNeighborsClassifier(n_neighbors=9)
predict_and_plot_confusion_matrix(train_x_responseCoding, train_y, cv_x_responseCoding, cv_y, clf)
```

    Log loss : 1.0688098269438249
    Number of mis-classified points : 0.37593984962406013
    -------------------- Confusion matrix --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_119_1.png)


    -------------------- Precision matrix (Columm Sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_119_3.png)


    -------------------- Recall matrix (Row sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_119_5.png)


# Logistic Regression

using all features of course.


```python
alpha = [10 ** x for x in range(-6, 3)]
cv_log_error_array = []
for i in alpha:
    print("for alpha =", i)
    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)
    #note the class-wight is balanced!
    clf.fit(train_x_onehotCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_onehotCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    # to avoid rounding error while multiplying probabilites we use log-probability estimates
    print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 
```

    for alpha = 1e-06
    Log Loss : 1.3335535183617702
    for alpha = 1e-05
    Log Loss : 1.3485060101243975
    for alpha = 0.0001
    Log Loss : 1.2663136562274235
    for alpha = 0.001
    Log Loss : 1.0616253515307865
    for alpha = 0.01
    Log Loss : 1.1156475597846707
    for alpha = 0.1
    Log Loss : 1.4478644889194012
    for alpha = 1
    Log Loss : 1.6814775680999903
    for alpha = 10
    Log Loss : 1.7083195165820848
    for alpha = 100
    Log Loss : 1.7111513005601562



```python
fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()
```


![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_122_0.png)



```python
best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

predict_y = sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
```

    For values of best alpha =  0.001 The train log loss is: 0.6354227595948588
    For values of best alpha =  0.001 The cross validation log loss is: 1.0616253515307865
    For values of best alpha =  0.001 The test log loss is: 1.1036317241204654



```python
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
```

    Log loss : 1.0616253515307865
    Number of mis-classified points : 0.3383458646616541
    -------------------- Confusion matrix --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_124_1.png)


    -------------------- Precision matrix (Columm Sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_124_3.png)


    -------------------- Recall matrix (Row sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_124_5.png)



```python
def get_imp_feature_names(text, indices, removed_ind = []):
    word_present = 0
    tabulte_list = []
    incresingorder_ind = 0
    for i in indices:
        if i < train_gene_feature_onehotCoding.shape[1]:
            tabulte_list.append([incresingorder_ind, "Gene", "Yes"])
        elif i< 18:
            tabulte_list.append([incresingorder_ind,"Variation", "Yes"])
        if ((i > 17) & (i not in removed_ind)) :
            word = train_text_features[i]
            yes_no = True if word in text.split() else False
            if yes_no:
                word_present += 1
            tabulte_list.append([incresingorder_ind,train_text_features[i], yes_no])
        incresingorder_ind += 1
    print(word_present, "most importent features are present in our query point")
    print("-"*50)
    print("The features that are most importent of the ",predicted_cls[0]," class:")
    print (tabulate(tabulte_list, headers=["Index",'Feature name', 'Present or Not']))
```


```python
# from tabulate import tabulate
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_x_onehotCoding,train_y)
test_point_index = 1
no_feature = 500
predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :", predicted_cls[0])
print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))
print("Actual Class :", test_y[test_point_index])
indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]
print("-"*50)
get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
```

    Predicted Class : 7
    Predicted Class Probabilities: [[1.500e-03 3.204e-01 1.500e-03 5.000e-03 2.200e-03 5.000e-04 6.570e-01
      7.300e-03 4.600e-03]]
    Actual Class : 7
    --------------------------------------------------
    28 Text feature [constitutive] present in test data point [True]
    42 Text feature [transforming] present in test data point [True]
    54 Text feature [thyroid] present in test data point [True]
    73 Text feature [constitutively] present in test data point [True]
    78 Text feature [activated] present in test data point [True]
    112 Text feature [murine] present in test data point [True]
    113 Text feature [transducers] present in test data point [True]
    134 Text feature [egf] present in test data point [True]
    138 Text feature [tk] present in test data point [True]
    155 Text feature [frederick] present in test data point [True]
    167 Text feature [downstream] present in test data point [True]
    179 Text feature [activation] present in test data point [True]
    182 Text feature [transformation] present in test data point [True]
    206 Text feature [filters] present in test data point [True]
    208 Text feature [cysteine] present in test data point [True]
    209 Text feature [expressing] present in test data point [True]
    233 Text feature [hyperplasia] present in test data point [True]
    279 Text feature [electroporated] present in test data point [True]
    281 Text feature [oncogene] present in test data point [True]
    309 Text feature [cdnas] present in test data point [True]
    325 Text feature [ligand] present in test data point [True]
    379 Text feature [3t3] present in test data point [True]
    386 Text feature [inhibited] present in test data point [True]
    411 Text feature [proliferate] present in test data point [True]
    431 Text feature [transformed] present in test data point [True]
    484 Text feature [juxtamembrane] present in test data point [True]
    497 Text feature [distinctly] present in test data point [True]
    Out of the top  500  features  27 are present in query point


# Linear Support Vector Machines


```python
alpha = [10 ** x for x in range(-5, 3)]
cv_log_error_array = []
for i in alpha:
    print("for C =", i)
#     clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')
    clf = SGDClassifier( class_weight='balanced', alpha=i, penalty='l2', loss='hinge', random_state=42)
    clf.fit(train_x_onehotCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_onehotCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 

fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()


best_alpha = np.argmin(cv_log_error_array)
# clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

predict_y = sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
```

    for C = 1e-05
    Log Loss : 1.3322173578538714
    for C = 0.0001
    Log Loss : 1.2914742159137906
    for C = 0.001
    Log Loss : 1.1741215303020727
    for C = 0.01
    Log Loss : 1.1023764445906772
    for C = 0.1
    Log Loss : 1.3514175353616704
    for C = 1
    Log Loss : 1.701214840749554
    for C = 10
    Log Loss : 1.711713564464156
    for C = 100
    Log Loss : 1.7117136745710113



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_128_1.png)


    For values of best alpha =  0.01 The train log loss is: 0.7470780675717899
    For values of best alpha =  0.01 The cross validation log loss is: 1.1023764445906772
    For values of best alpha =  0.01 The test log loss is: 1.147745555306825



```python
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42,class_weight='balanced')
predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,cv_y, clf)
```

    Log loss : 1.1023764445906772
    Number of mis-classified points : 0.36278195488721804
    -------------------- Confusion matrix --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_129_1.png)


    -------------------- Precision matrix (Columm Sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_129_3.png)


    -------------------- Recall matrix (Row sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_129_5.png)



```python
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)
clf.fit(train_x_onehotCoding,train_y)
test_point_index = 50
# test_point_index = 100
no_feature = 500
predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :", predicted_cls[0])
print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))
print("Actual Class :", test_y[test_point_index])
indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]
print("-"*50)
get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
```

    Predicted Class : 4
    Predicted Class Probabilities: [[0.3201 0.0331 0.0104 0.4531 0.0452 0.0157 0.1112 0.0074 0.0038]]
    Actual Class : 1
    --------------------------------------------------
    20 Text feature [suppressor] present in test data point [True]
    37 Text feature [brachmann] present in test data point [True]
    112 Text feature [instability] present in test data point [True]
    115 Text feature [proteasome] present in test data point [True]
    164 Text feature [n239y] present in test data point [True]
    169 Text feature [unstable] present in test data point [True]
    245 Text feature [satisfactorily] present in test data point [True]
    272 Text feature [lys] present in test data point [True]
    337 Text feature [degradation] present in test data point [True]
    342 Text feature [superposition] present in test data point [True]
    351 Text feature [h179r] present in test data point [True]
    441 Text feature [compromised] present in test data point [True]
    468 Text feature [stability] present in test data point [True]
    Out of the top  500  features  13 are present in query point


# Random Forest Classifier


```python
# with one-hot encoding
alpha = [100,200,500,1000,2000]
max_depth = [5, 10]
cv_log_error_array = []
for i in alpha:
    for j in max_depth:
        print("for n_estimators =", i,"and max depth = ", j)
        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)
        clf.fit(train_x_onehotCoding, train_y)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_x_onehotCoding, train_y)
        sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
        print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 

```

    for n_estimators = 100 and max depth =  5
    Log Loss : 1.2357079747547164
    for n_estimators = 100 and max depth =  10
    Log Loss : 1.171679037791546
    for n_estimators = 200 and max depth =  5
    Log Loss : 1.2280462370539889
    for n_estimators = 200 and max depth =  10
    Log Loss : 1.1610685007427581
    for n_estimators = 500 and max depth =  5
    Log Loss : 1.2165988170456992
    for n_estimators = 500 and max depth =  10
    Log Loss : 1.1594151282109517
    for n_estimators = 1000 and max depth =  5
    Log Loss : 1.212515668739816
    for n_estimators = 1000 and max depth =  10
    Log Loss : 1.1568610186006922
    for n_estimators = 2000 and max depth =  5
    Log Loss : 1.2119707883790425
    for n_estimators = 2000 and max depth =  10
    Log Loss : 1.1551986614674263



```python
best_alpha = np.argmin(cv_log_error_array)
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

predict_y = sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best estimator = ', alpha[int(best_alpha/2)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best estimator = ', alpha[int(best_alpha/2)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best estimator = ', alpha[int(best_alpha/2)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
```

    For values of best estimator =  2000 The train log loss is: 0.709581415667774
    For values of best estimator =  2000 The cross validation log loss is: 1.1551986614674263
    For values of best estimator =  2000 The test log loss is: 1.1270622147289093



```python
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)
predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,cv_y, clf)
```

    Log loss : 1.1551986614674263
    Number of mis-classified points : 0.3890977443609023
    -------------------- Confusion matrix --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_134_1.png)


    -------------------- Precision matrix (Columm Sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_134_3.png)


    -------------------- Recall matrix (Row sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_134_5.png)



```python
# test_point_index = 10
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

test_point_index = 1
no_feature = 100
predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :", predicted_cls[0])
print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))
print("Actual Class :", test_y[test_point_index])
indices = np.argsort(-clf.feature_importances_)
print("-"*50)
get_impfeature_names(indices[:no_feature], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
```

    Predicted Class : 7
    Predicted Class Probabilities: [[0.0742 0.1345 0.0141 0.0758 0.0401 0.0371 0.6132 0.0051 0.006 ]]
    Actual Class : 7
    --------------------------------------------------
    0 Text feature [kinase] present in test data point [True]
    1 Text feature [activating] present in test data point [True]
    2 Text feature [tyrosine] present in test data point [True]
    3 Text feature [constitutive] present in test data point [True]
    4 Text feature [treatment] present in test data point [True]
    5 Text feature [inhibitors] present in test data point [True]
    6 Text feature [activated] present in test data point [True]
    7 Text feature [missense] present in test data point [True]
    8 Text feature [activation] present in test data point [True]
    9 Text feature [inhibitor] present in test data point [True]
    10 Text feature [phosphorylation] present in test data point [True]
    11 Text feature [suppressor] present in test data point [True]
    12 Text feature [function] present in test data point [True]
    13 Text feature [constitutively] present in test data point [True]
    14 Text feature [erk] present in test data point [True]
    15 Text feature [signaling] present in test data point [True]
    16 Text feature [functional] present in test data point [True]
    17 Text feature [months] present in test data point [True]
    19 Text feature [transforming] present in test data point [True]
    20 Text feature [receptor] present in test data point [True]
    24 Text feature [oncogenic] present in test data point [True]
    28 Text feature [growth] present in test data point [True]
    31 Text feature [loss] present in test data point [True]
    33 Text feature [extracellular] present in test data point [True]
    34 Text feature [kinases] present in test data point [True]
    35 Text feature [inhibition] present in test data point [True]
    37 Text feature [activate] present in test data point [True]
    38 Text feature [cells] present in test data point [True]
    42 Text feature [patients] present in test data point [True]
    47 Text feature [resistance] present in test data point [True]
    50 Text feature [downstream] present in test data point [True]
    51 Text feature [3t3] present in test data point [True]
    53 Text feature [cell] present in test data point [True]
    55 Text feature [erk1] present in test data point [True]
    56 Text feature [efficacy] present in test data point [True]
    60 Text feature [proliferation] present in test data point [True]
    62 Text feature [treated] present in test data point [True]
    65 Text feature [protein] present in test data point [True]
    68 Text feature [expressing] present in test data point [True]
    69 Text feature [p53] present in test data point [True]
    71 Text feature [ligand] present in test data point [True]
    73 Text feature [lines] present in test data point [True]
    74 Text feature [clinical] present in test data point [True]
    76 Text feature [autophosphorylation] present in test data point [True]
    77 Text feature [resistant] present in test data point [True]
    80 Text feature [harboring] present in test data point [True]
    81 Text feature [il] present in test data point [True]
    88 Text feature [mutant] present in test data point [True]
    89 Text feature [stimulation] present in test data point [True]
    90 Text feature [serum] present in test data point [True]
    94 Text feature [assays] present in test data point [True]
    96 Text feature [proteins] present in test data point [True]
    Out of the top  100  features  52 are present in query point


# Stack Models together

We have Logistic Regression + Linear SVM + NB


```python
clf1 = SGDClassifier(alpha=0.001, penalty='l2', loss='log', class_weight='balanced', random_state=0)
clf1.fit(train_x_onehotCoding, train_y)
sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")

clf2 = SGDClassifier(alpha=1, penalty='l2', loss='hinge', class_weight='balanced', random_state=0)
clf2.fit(train_x_onehotCoding, train_y)
sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")


clf3 = MultinomialNB(alpha=0.001)
clf3.fit(train_x_onehotCoding, train_y)
sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")

sig_clf1.fit(train_x_onehotCoding, train_y)
print("Logistic Regression :  Log Loss: %0.2f" % (log_loss(cv_y, sig_clf1.predict_proba(cv_x_onehotCoding))))
sig_clf2.fit(train_x_onehotCoding, train_y)
print("Support vector machines : Log Loss: %0.2f" % (log_loss(cv_y, sig_clf2.predict_proba(cv_x_onehotCoding))))
sig_clf3.fit(train_x_onehotCoding, train_y)
print("Naive Bayes : Log Loss: %0.2f" % (log_loss(cv_y, sig_clf3.predict_proba(cv_x_onehotCoding))))
print("-"*50)
alpha = [0.0001,0.001,0.01,0.1,1,10] 
best_alpha = 999
for i in alpha:
    lr = LogisticRegression(C=i)
    sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)
    sclf.fit(train_x_onehotCoding, train_y)
    print("Stacking Classifer : for the value of alpha: %f Log Loss: %0.3f" % (i, log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))))
    log_error =log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))
    if best_alpha > log_error:
        best_alpha = log_error
        
# hyperparameter is an inverse measure of regularization, smaller then stronger regularization
```

    Logistic Regression :  Log Loss: 1.08
    Support vector machines : Log Loss: 1.70
    Naive Bayes : Log Loss: 1.25
    --------------------------------------------------
    Stacking Classifer : for the value of alpha: 0.000100 Log Loss: 2.179
    Stacking Classifer : for the value of alpha: 0.001000 Log Loss: 2.043
    Stacking Classifer : for the value of alpha: 0.010000 Log Loss: 1.535
    Stacking Classifer : for the value of alpha: 0.100000 Log Loss: 1.116
    Stacking Classifer : for the value of alpha: 1.000000 Log Loss: 1.159
    Stacking Classifer : for the value of alpha: 10.000000 Log Loss: 1.385



```python
lr = LogisticRegression(C=0.1)
sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)
sclf.fit(train_x_onehotCoding, train_y)

log_error = log_loss(train_y, sclf.predict_proba(train_x_onehotCoding))
print("Log loss (train) on the stacking classifier :",log_error)

log_error = log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))
print("Log loss (CV) on the stacking classifier :",log_error)

log_error = log_loss(test_y, sclf.predict_proba(test_x_onehotCoding))
print("Log loss (test) on the stacking classifier :",log_error)

print("Number of missclassified point :", np.count_nonzero((sclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0])
plot_confusion_matrix(test_y=test_y, predict_y=sclf.predict(test_x_onehotCoding))
```

    Log loss (train) on the stacking classifier : 0.6836347781108095
    Log loss (CV) on the stacking classifier : 1.1158703751666934
    Log loss (test) on the stacking classifier : 1.189259947803605
    Number of missclassified point : 0.3924812030075188
    -------------------- Confusion matrix --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_138_1.png)


    -------------------- Precision matrix (Columm Sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_138_3.png)


    -------------------- Recall matrix (Row sum=1) --------------------



![png](Cancer_Treatment_EDA_Train_files/Cancer_Treatment_EDA_Train_138_5.png)

