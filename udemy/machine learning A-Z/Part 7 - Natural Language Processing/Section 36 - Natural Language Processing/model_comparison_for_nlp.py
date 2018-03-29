# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from metric import Evaluations

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
stopwordSet = set(stopwords.words('english'))
for i in range(0, 1000):
    # Change non letter character to space
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwordSet]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)  # X and y already scalled by CountVectorizer

# Prepare plot variables
numModels = 6
numMetrics = 4
models = ['Naive Bayes', 'Logistic Regression', 'K-NN', 'Linear SVM', 'CART', 'Random Forest']
modelMetrics = np.zeros([numMetrics, numModels])  # accuracy, precission, recall, f1


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Since this is binary case, then compute tn, fp, fn, and tp can be done like this:
evaluations = Evaluations(confusion_matrix(y_test, y_pred))
modelMetrics[0][0] = evaluations.accuracy()
modelMetrics[1][0] = evaluations.precission()
modelMetrics[2][0] = evaluations.recall()
modelMetrics[3][0] = evaluations.f1()

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Since this is binary case, then compute tn, fp, fn, and tp can be done like this:
evaluations = Evaluations(confusion_matrix(y_test, y_pred))
modelMetrics[0][1] = evaluations.accuracy()
modelMetrics[1][1] = evaluations.precission()
modelMetrics[2][1] = evaluations.recall()
modelMetrics[3][1] = evaluations.f1()

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Since this is binary case, then compute tn, fp, fn, and tp can be done like this:
evaluations = Evaluations(confusion_matrix(y_test, y_pred))
modelMetrics[0][2] = evaluations.accuracy()
modelMetrics[1][2] = evaluations.precission()
modelMetrics[2][2] = evaluations.recall()
modelMetrics[3][2] = evaluations.f1()

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
#classifier = SVC(kernel = 'poly', degree = 6, random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Since this is binary case, then compute tn, fp, fn, and tp can be done like this:
evaluations = Evaluations(confusion_matrix(y_test, y_pred))
modelMetrics[0][3] = evaluations.accuracy()
modelMetrics[1][3] = evaluations.precission()
modelMetrics[2][3] = evaluations.recall()
modelMetrics[3][3] = evaluations.f1()

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Since this is binary case, then compute tn, fp, fn, and tp can be done like this:
evaluations = Evaluations(confusion_matrix(y_test, y_pred))
modelMetrics[0][4] = evaluations.accuracy()
modelMetrics[1][4] = evaluations.precission()
modelMetrics[2][4] = evaluations.recall()
modelMetrics[3][4] = evaluations.f1()

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Since this is binary case, then compute tn, fp, fn, and tp can be done like this:
evaluations = Evaluations(confusion_matrix(y_test, y_pred))
modelMetrics[0][5] = evaluations.accuracy()
modelMetrics[1][5] = evaluations.precission()
modelMetrics[2][5] = evaluations.recall()
modelMetrics[3][5] = evaluations.f1()


# Transform modelMetrics into a DataFrame
df = pd.DataFrame(data=modelMetrics,
                  index=['Accuracy', 'Precission', 'Recall', 'F1 Score'],
                  columns=pd.Index(models, name='Models'))

# Inspect the DataFrame
# print(df)

# Save DataFrame as csv file
df.to_csv('confusion-matrix.csv', sep=',')

# Plot the accuracy, precission, recall, and f1 score for each model
df.plot(kind='bar', colormap='Greys', grid=True, rot=0)
plt.style.use('ggplot')
plt.legend(loc='upper center', bbox_to_anchor=(
    0.5, -0.05), shadow=True, ncol=numModels)
f = plt.figure(1)
extra_args = dict(family='serif', ha='center',
                  va='top', transform=f.transFigure)
f.text(.5, .99,
       'Comparisson of Classification Model Performances in Binary Sentiment Analysis (Like or Dislike)',
       size=12, **extra_args)
f.text(.5, .96,
       'Using small dataset contains 1000 user reviews with like/dislike written in English.',
       size=9, **extra_args)  # Subtitle
extra_args.update(ha='left', va='bottom', size=10, ma='right')
f.text(0.65, 0.0,
       'Connect with me at: linkedin.com/in/hafidz-jazuli-luthfi', **extra_args)  # Caption
f.canvas.draw()
plt.show()
