# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Prepare columns of datasets
metrics = c()
models = c()
values = c()

# Fitting Logistic Regression to the Training set
classifier = glm(formula = Liked ~ .,
                 family = binomial,
                 data = training_set)
# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-692])
y_pred = ifelse(prob_pred > 0.5, 1, 0)
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_test_numeric = as.numeric(test_set$Liked == 1)
y_pred_numeric = as.numeric(y_pred == 1)
tp = sum(y_test_numeric & y_pred_numeric)
accuracy = sum(!(xor(y_test_numeric, y_pred_numeric))) / length(y_pred_numeric)
precision = tp / sum(y_pred_numeric)
recall = tp / sum(y_test_numeric)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
# Adding each evaluation metric into metrics vector
metrics[length(metrics)+1] = 'Accuracy'
models[length(models)+1] = 'Logistic Regression'
values[length(values)+1] = accuracy
metrics[length(metrics)+1] = 'Precision'
models[length(models)+1] = 'Logistic Regression'
values[length(values)+1] = precision
metrics[length(metrics)+1] = 'Recall'
models[length(models)+1] = 'Logistic Regression'
values[length(values)+1] = recall
metrics[length(metrics)+1] = 'F1 Score'
models[length(models)+1] = 'Logistic Regression'
values[length(values)+1] = f1_score

# Fitting Maximum Entropy to the Training set
# install.packages('maxent')
library(maxent)
classifier = maxent(feature_matrix = training_set[-692],
                    code_vector = training_set$Liked)
# Predicting the Test set results
y_pred = predict(classifier, test_set[-692])
y_pred = y_pred[1:length(test_set$Liked)]
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_test_numeric = as.numeric(test_set$Liked == 1)
y_pred_numeric = as.numeric(y_pred == 1)
tp = sum(y_test_numeric & y_pred_numeric)
accuracy = sum(!(xor(y_test_numeric, y_pred_numeric))) / length(y_pred_numeric)
precision = tp / sum(y_pred_numeric)
recall = tp / sum(y_test_numeric)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
# Adding each evaluation metric into metrics vector
metrics[length(metrics)+1] = 'Accuracy'
models[length(models)+1] = 'Maximum Entropy'
values[length(values)+1] = accuracy
metrics[length(metrics)+1] = 'Precision'
models[length(models)+1] = 'Maximum Entropy'
values[length(values)+1] = precision
metrics[length(metrics)+1] = 'Recall'
models[length(models)+1] = 'Maximum Entropy'
values[length(values)+1] = recall
metrics[length(metrics)+1] = 'F1 Score'
models[length(models)+1] = 'Maximum Entropy'
values[length(values)+1] = f1_score

# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
y_pred = knn(train = training_set[, -692],
             test = test_set[, -692],
             cl = training_set$Liked,
             k = 5,
             prob = TRUE)
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_test_numeric = as.numeric(test_set$Liked == 1)
y_pred_numeric = as.numeric(y_pred == 1)
tp = sum(y_test_numeric & y_pred_numeric)
accuracy = sum(!(xor(y_test_numeric, y_pred_numeric))) / length(y_pred_numeric)
precision = tp / sum(y_pred_numeric)
recall = tp / sum(y_test_numeric)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
# Adding each evaluation metric into metrics vector
metrics[length(metrics)+1] = 'Accuracy'
models[length(models)+1] = 'K-NN (k=5)'
values[length(values)+1] = accuracy
metrics[length(metrics)+1] = 'Precision'
models[length(models)+1] = 'K-NN (k=5)'
values[length(values)+1] = precision
metrics[length(metrics)+1] = 'Recall'
models[length(models)+1] = 'K-NN (k=5)'
values[length(values)+1] = recall
metrics[length(metrics)+1] = 'F1 Score'
models[length(models)+1] = 'K-NN (k=5)'
values[length(values)+1] = f1_score

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_test_numeric = as.numeric(test_set$Liked == 1)
y_pred_numeric = as.numeric(y_pred == 1)
tp = sum(y_test_numeric & y_pred_numeric)
accuracy = sum(!(xor(y_test_numeric, y_pred_numeric))) / length(y_pred_numeric)
precision = tp / sum(y_pred_numeric)
recall = tp / sum(y_test_numeric)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
# Adding each evaluation metric into metrics vector
metrics[length(metrics)+1] = 'Accuracy'
models[length(models)+1] = 'SVM Linear'
values[length(values)+1] = accuracy
metrics[length(metrics)+1] = 'Precision'
models[length(models)+1] = 'SVM Linear'
values[length(values)+1] = precision
metrics[length(metrics)+1] = 'Recall'
models[length(models)+1] = 'SVM Linear'
values[length(values)+1] = recall
metrics[length(metrics)+1] = 'F1 Score'
models[length(models)+1] = 'SVM Linear'
values[length(values)+1] = f1_score

# Fitting Decision Tree Classification to the Training set
# install.packages('rpart')
library(rpart)
classifier = rpart(formula = Liked ~ .,
                   data = training_set)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_test_numeric = as.numeric(test_set$Liked == 1)
y_pred_numeric = as.numeric(y_pred == 1)
tp = sum(y_test_numeric & y_pred_numeric)
accuracy = sum(!(xor(y_test_numeric, y_pred_numeric))) / length(y_pred_numeric)
precision = tp / sum(y_pred_numeric)
recall = tp / sum(y_test_numeric)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
# Adding each evaluation metric into metrics vector
metrics[length(metrics)+1] = 'Accuracy'
models[length(models)+1] = 'CART'
values[length(values)+1] = accuracy
metrics[length(metrics)+1] = 'Precision'
models[length(models)+1] = 'CART'
values[length(values)+1] = precision
metrics[length(metrics)+1] = 'Recall'
models[length(models)+1] = 'CART'
values[length(values)+1] = recall
metrics[length(metrics)+1] = 'F1 Score'
models[length(models)+1] = 'CART'
values[length(values)+1] = f1_score

# Apply C50 algorithm on training set
# install.packages('C50')
library(C50)
classifier = C5.0(formula = Liked ~ ., data = training_set)
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_test_numeric = as.numeric(test_set$Liked == 1)
y_pred_numeric = as.numeric(y_pred == 1)
tp = sum(y_test_numeric & y_pred_numeric)
accuracy = sum(!(xor(y_test_numeric, y_pred_numeric))) / length(y_pred_numeric)
precision = tp / sum(y_pred_numeric)
recall = tp / sum(y_test_numeric)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
# Adding each evaluation metric into metrics vector
metrics[length(metrics)+1] = 'Accuracy'
models[length(models)+1] = 'C50'
values[length(values)+1] = accuracy
metrics[length(metrics)+1] = 'Precision'
models[length(models)+1] = 'C50'
values[length(values)+1] = precision
metrics[length(metrics)+1] = 'Recall'
models[length(models)+1] = 'C50'
values[length(values)+1] = recall
metrics[length(metrics)+1] = 'F1 Score'
models[length(models)+1] = 'C50'
values[length(values)+1] = f1_score

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])
# Making the Confusion Matrix
# cm = table(test_set[, 692], y_pred)
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_test_numeric = as.numeric(test_set$Liked == 1)
y_pred_numeric = as.numeric(y_pred == 1)
tp = sum(y_test_numeric & y_pred_numeric)
accuracy = sum(!(xor(y_test_numeric, y_pred_numeric))) / length(y_pred_numeric)
precision = tp / sum(y_pred_numeric)
recall = tp / sum(y_test_numeric)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
# Adding each evaluation metric into metrics vector
metrics[length(metrics)+1] = 'Accuracy'
models[length(models)+1] = 'Random Forest (tree=10)'
values[length(values)+1] = accuracy
metrics[length(metrics)+1] = 'Precision'
models[length(models)+1] = 'Random Forest (tree=10)'
values[length(values)+1] = precision
metrics[length(metrics)+1] = 'Recall'
models[length(models)+1] = 'Random Forest (tree=10)'
values[length(values)+1] = recall
metrics[length(metrics)+1] = 'F1 Score'
models[length(models)+1] = 'Random Forest (tree=10)'
values[length(values)+1] = f1_score

# Fitting Naive Bayes to the Training set
# install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-692],
                        y = training_set$Liked)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])
# Making the Confusion Matrix
# cm = table(test_set[, 692], y_pred)
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_test_numeric = as.numeric(test_set$Liked == 1)
y_pred_numeric = as.numeric(y_pred == 1)
tp = sum(y_test_numeric & y_pred_numeric)
accuracy = sum(!(xor(y_test_numeric, y_pred_numeric))) / length(y_pred_numeric)
precision = tp / sum(y_pred_numeric)
recall = tp / sum(y_test_numeric)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
# Adding each evaluation metric into metrics vector
metrics[length(metrics)+1] = 'Accuracy'
models[length(models)+1] = 'Naive Bayes'
values[length(values)+1] = accuracy
metrics[length(metrics)+1] = 'Precision'
models[length(models)+1] = 'Naive Bayes'
values[length(values)+1] = precision
metrics[length(metrics)+1] = 'Recall'
models[length(models)+1] = 'Naive Bayes'
values[length(values)+1] = recall
metrics[length(metrics)+1] = 'F1 Score'
models[length(models)+1] = 'Naive Bayes'
values[length(values)+1] = f1_score


# Plot each model measurement
modelMetrics = data.frame(metrics, models, values)
plot = ggplot(modelMetrics, aes(metrics, values, fill=models)) +
  geom_bar(stat = 'identity', position="dodge")  +
  scale_fill_grey(start=0.8, end=0.2) +
  theme_classic()
plot = plot + labs(title="Comparisson of Classification Model Performances in Binary Sentiment Analysis (Like or Dislike)",
            subtitle="Using small dataset contains 1000 user reviews with like/dislike written in English.",
            caption="Connect with me at: linkedin.com/in/hafidz-jazuli-luthfi")