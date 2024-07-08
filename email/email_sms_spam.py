import pandas as pd 
import numpy as np
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.head(5)
df.info()

# drop last 3 cols
df.drop(columns = ['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'], inplace=True)

df.head()
#renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.head(5)
from sklearn.preprocessing import LabelEncoder
encoder =LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
df.head()
# missing value
df.isnull().sum()
#check the duplicate value
df.duplicated().sum()
#remove the duplicate value
df=df.drop_duplicates(keep='first')
df.duplicated().sum()
df.shape
#HOW MANY SPAM DATA AND HOW MANY NON SPAM DATA
df['target'].value_counts()
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['Non-spam','spam'],autopct="%0.2f")
plt.show()
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.20, random_state=42)

# Vectorize the text data
cv = CountVectorizer()
X_train_vect = cv.fit_transform(X_train)
X_test_vect = cv.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Predict on the test data
y_pred = model.predict(X_test_vect)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Vectorize new email message data
new_messages = ['10 doller gift from apple','win the 1 lakh dollor from the google','hi manish ']
new_messages_vect = cv.transform(new_messages)

# Predict labels of new email message data
new_predictions = model.predict(new_messages_vect)

# Print predicted labels
print(new_predictions)

import pickle
pickle.dump(model,open("spam.pkl","wb"))
pickle.dump(cv,open("vectorizer.pkl","wb"))
clf=pickle.load(open("spam.pkl","rb"))
clf