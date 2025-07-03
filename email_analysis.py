#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


data = pd.read_csv("C:/Users/Dharshan/OneDrive/Desktop/email/Phishing_Email.csv")
data


# In[7]:


data.shape


# In[9]:


safe_email = data[data["Email Type"] == "Safe Email"]
safe_cnt = safe_email["Email Type"].count()
print(f"The total number of safe email is {safe_cnt}")


# In[11]:


phishing_email = data[data["Email Type"] == "Phishing Email"]
phishing_cnt = phishing_email["Email Type"].count()
print(f"The total number of safe email is {phishing_cnt}")


# In[13]:


data.isnull().sum()


# In[15]:


data.drop(["Unnamed: 0"],axis=1,inplace=True)
data.dropna(axis=0,inplace=True)
data.drop_duplicates(inplace=True)


# In[17]:


print("Dimension of the row data:",data.shape)


# In[19]:


data['Email Type'].value_counts().values


# In[21]:


import plotly.express as px
fig = px.bar(data['Email Type'].value_counts(), x=data['Email Type'].value_counts().index, y=data['Email Type'].value_counts().values,
             color=['blue', 'red'], labels={'x': 'Category', 'y': 'Count'},
             title="Categorical Distribution")


#fig.show()


# In[23]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data["Email Type"] = LE.fit_transform(data["Email Type"])


# In[25]:


# Phishing = 0
# Safe = 1

data.head()


# In[29]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
safe_email = data[data["Email Type"] == 1]
safe_text = " ".join(safe_email["Email Text"])
word_cloud = WordCloud(width=800,height=400,background_color="white").generate(safe_text)
plt.figure(figsize=(10,6))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis("off")
#plt.show()


# In[31]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
phishing_email = data[data["Email Type"] == 0]
phishing_text = " ".join(phishing_email["Email Text"])
word_cloud = WordCloud(width=800,height=400,background_color="white").generate(phishing_text)
plt.figure(figsize=(10,6))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis("off")
#plt.show()


# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    max_df=0.9
)
x_feature = vectorizer.fit_transform(data["Email Text"]).toarray()


# In[34]:


x = x_feature
y = np.array(data['Email Type'])


# In[39]:



import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[41]:


model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# In[43]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[45]:


model.fit(X_train, y_train, epochs=10, batch_size=4)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")


# In[47]:


from sklearn.metrics import classification_report
import numpy as np


y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)


print(classification_report(y_test, y_pred, target_names=['Not Phishing', 'Phishing']))


# In[49]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Phishing', 'Phishing'], yticklabels=['Not Phishing', 'Phishing'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
#plt.show()


# In[51]:


from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)


plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
#plt.show()


# In[53]:


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
#plt.show()


# In[55]:


print(LE.classes_)


# In[57]:


new_emails = [
    "Click here to claim your prize!",
    "Your invoice is attached. Please review it.",
    "Urgent: Verify your account immediately.",
]

new_X = vectorizer.transform(new_emails).toarray()

predictions = model.predict(new_X)

predicted_labels = (predictions > 0.5).astype(int)

for email, label in zip(new_emails, predicted_labels.flatten()):

    predicted_class = 'phishing' if label == 0 else 'legitimate'
    print(f"Email: {email}\nPredicted label: {predicted_class}\n")

import pickle

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save model (TensorFlow format)
model.save("phishing_model.keras")

