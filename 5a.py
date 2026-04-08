# NAIVE BAYES - ALL APPLICATIONS IN ONE PROGRAM

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

print("\n===== 1. EMAIL SPAM FILTERING =====")
emails = [
    "Win money now", "Limited offer claim prize",
    "Meeting schedule tomorrow", "Project discussion today",
    "Congratulations you won lottery", "Please review the report"
]
labels = ["spam", "spam", "ham", "ham", "spam", "ham"]

cv = CountVectorizer()
X = cv.fit_transform(emails)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=0)

model1 = MultinomialNB()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
print("Spam Accuracy:", accuracy_score(y_test, pred1))


print("\n===== 2. MEDICAL DIAGNOSIS =====")
data_medical = pd.DataFrame({
    'Fever': [1,1,0,1,0,0,1,0],
    'Cough': [1,0,1,1,0,1,1,0],
    'Headache': [1,1,0,1,0,0,1,0],
    'Flu': [1,1,0,1,0,0,1,0]
})

X = data_medical[['Fever','Cough','Headache']]
y = data_medical['Flu']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model2 = CategoricalNB()
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)
print("Medical Diagnosis Accuracy:", accuracy_score(y_test, pred2))


print("\n===== 3. SENTIMENT ANALYSIS =====")
reviews = [
    "This movie is amazing", "I hate this film",
    "Fantastic acting and story", "Worst movie ever",
    "I love this", "Terrible experience"
]
sentiment = ["positive", "negative", "positive", "negative", "positive", "negative"]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(reviews)

X_train, X_test, y_train, y_test = train_test_split(X, sentiment, test_size=0.3, random_state=0)

model3 = MultinomialNB()
model3.fit(X_train, y_train)
pred3 = model3.predict(X_test)
print("Sentiment Analysis Accuracy:", accuracy_score(y_test, pred3))


print("\n===== 4. PRODUCT RECOMMENDATION =====")
# Features: [time_spent, pages_viewed]
X = np.array([[5,10],[10,20],[2,5],[8,15],[1,3],[12,25]])
y = ["buy","buy","not_buy","buy","not_buy","buy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model4 = GaussianNB()
model4.fit(X_train, y_train)
pred4 = model4.predict(X_test)
print("Product Recommendation Accuracy:", accuracy_score(y_test, pred4))


print("\n===== 5. WEATHER PREDICTION (PLAY TENNIS) =====")
data_weather = pd.DataFrame({
    'Outlook':[0,1,2,0,2,1],
    'Temp':[0,1,1,0,1,0],
    'Humidity':[1,1,0,1,0,0],
    'Windy':[0,1,0,0,1,1],
    'Play':[0,1,1,0,1,1]
})

X = data_weather[['Outlook','Temp','Humidity','Windy']]
y = data_weather['Play']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model5 = CategoricalNB()
model5.fit(X_train, y_train)
pred5 = model5.predict(X_test)
print("Weather Prediction Accuracy:", accuracy_score(y_test, pred5))
