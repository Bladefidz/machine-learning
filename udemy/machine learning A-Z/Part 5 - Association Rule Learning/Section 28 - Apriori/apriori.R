# Apriori

# Data Preprocessing
# dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
# Use sparse matrix instead
# install.packages('arules')
library(arules)  # Load sparse matrix module
# Dataset contains transaction in a week
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
# Minimum support is 4% or (4*7)/7501 or at least four transactions in a week
# Minimum confidence is 20%
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])