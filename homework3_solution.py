import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

# Some data structures that will be useful

allReviews = []
for l in readGz("train.json.gz"):
  allReviews.append(l)

reviewsTrain = allReviews[:100000]
reviewsValid = allReviews[100000:]
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
for r in reviewsTrain:
  reviewsPerUser[r['userID']].append(r)
  reviewsPerItem[r['businessID']].append(r)

##################################################
# Rating prediction (CSE258 only)                #
##################################################

trainRatings = [r['rating'] for r in reviewsTrain]
globalAverage = sum(trainRatings) * 1.0 / len(trainRatings)

validMSE = 0
for r in reviewsValid:
  se = (r['rating'] - globalAverage)**2
  validMSE += se

validMSE /= len(reviewsValid)

print("Validation MSE (average only) = " + str(validMSE))

betaU = {}
betaI = {}
for u in reviewsPerUser:
  betaU[u] = 0

for i in reviewsPerItem:
  betaI[i] = 0

alpha = globalAverage # Could initialize anywhere but this is a good guess

def iterate(lamb):
  newAlpha = 0
  for r in reviewsTrain:
    newAlpha += r['rating'] - (betaU[r['userID']] + betaI[r['businessID']])
  alpha = newAlpha / len(reviewsTrain)
  for u in reviewsPerUser:
    newBetaU = 0
    for r in reviewsPerUser[u]:
      newBetaU += r['rating'] - (alpha + betaI[r['businessID']])
    betaU[u] = newBetaU / (lamb + len(reviewsPerUser[u]))
  for i in reviewsPerItem:
    newBetaI = 0
    for r in reviewsPerItem[i]:
      newBetaI += r['rating'] - (alpha + betaU[r['userID']])
    betaI[i] = newBetaI / (lamb + len(reviewsPerItem[i]))
  mse = 0
  for r in reviewsTrain:
    prediction = alpha + betaU[r['userID']] + betaI[r['businessID']]
    mse += (r['rating'] - prediction)**2
  regularizer = 0
  for u in betaU:
    regularizer += betaU[u]**2
  for i in betaI:
    regularizer += betaI[i]**2
  mse /= len(reviewsTrain)
  return mse, mse + lamb*regularizer

# Fit with lambda = 1

mse,objective = iterate(1)
newMSE,newObjective = iterate(1)
iterations = 2

while iterations < 10 or objective - newObjective > 0.0001:
  mse, objective = newMSE, newObjective
  newMSE, newObjective = iterate(1)
  iterations += 1
  print("Objective after " + str(iterations) + " iterations = " + str(newObjective))
  print("MSE after " + str(iterations) + " iterations = " + str(newMSE))

validMSE = 0
for r in reviewsValid:
  bu = 0
  bi = 0
  if r['userID'] in betaU:
    bu = betaU[r['userID']]
  if r['businessID'] in betaI:
    bi = betaI[r['businessID']]
  prediction = alpha + bu + bi
  validMSE += (r['rating'] - prediction)**2

validMSE /= len(reviewsValid)
print("Validation MSE = " + str(validMSE))

betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()

print("Maximum betaU = " + str(betaUs[-1][1]) + ' (' + str(betaUs[-1][0]))
print("Maximum betaI = " + str(betaIs[-1][1]) + ' (' + str(betaIs[-1][0]))

# Better lambda...

iterations = 1
while iterations < 10 or objective - newObjective > 0.0001:
  mse, objective = newMSE, newObjective
  newMSE, newObjective = iterate(100)
  iterations += 1
  print("Objective after " + str(iterations) + " iterations = " + str(newObjective))
  print("MSE after " + str(iterations) + " iterations = " + str(newMSE))

validMSE = 0
for r in reviewsValid:
  bu = 0
  bi = 0
  if r['userID'] in betaU:
    bu = betaU[r['userID']]
  if r['businessID'] in betaI:
    bi = betaI[r['businessID']]
  prediction = alpha + bu + bi
  validMSE += (r['rating'] - prediction)**2

validMSE /= len(reviewsValid)
print("Validation MSE = " + str(validMSE))

# Format for kaggle

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  u,i = l.strip().split('-')
  bu = 0
  bi = 0
  if u in betaU:
    bu = betaU[u]
  if i in betaI:
    bi = betaI[i]
  _ = predictions.write(u + '-' + i + ',' + str(alpha + bu + bi) + '\n')

predictions.close()

##################################################
# Visit prediction                               #
##################################################

# Generate a negative set

userSet = set()
businessSet = set()
visitedSet = set()

for r in allReviews:
  userSet.add(r['userID'])
  businessSet.add(r['businessID'])
  visitedSet.add((r['userID'],r['businessID']))

lUserSet = list(userSet)
lBusinessSet = list(businessSet)

notVisited = set()
while (len(notVisited) < 100000):
  u = random.choice(lUserSet)
  b = random.choice(lBusinessSet)
  if (u,b) in visitedSet or (u,b) in notVisited: continue
  notVisited.add((u,b))

visitedValid = set()
for r in reviewsValid:
  visitedValid.add((r['userID'],r['businessID']))

# Collect the set of business categories visited by each user, and the set of categories to which a business belongs

visitedCategories = defaultdict(set)
businessCategories = defaultdict(set)

for r in reviewsTrain:
  if 'categories' in r:
    for c in r['categories']:
      visitedCategories[r['userID']].add(c)
      businessCategories[r['businessID']].add(c)

# Accuracy of our simple predictor

correct = 0
for (label,sample) in [(1, visitedValid), (0, notVisited)]:
  for (u,b) in sample:
    bc = set()
    uc = set()
    if b in businessCategories:
      bc = businessCategories[b]
    if u in visitedCategories:
      uc = visitedCategories[u]
    prediction = 0
    if len(bc.intersection(uc)): # User has visited some business of the same category
      prediction = 1
    if prediction == label:
      correct += 1

print("Validation accuracy (simple model) = " + str(correct * 1.0 / (len(visitedValid) + len(notVisited))))

predictions = open("predictions_Visit.txt", 'w')
for l in open("pairs_Visit.txt"):
  u,b = l.strip().split('-')
  bc = set()
  uc = set()
  if b in businessCategories:
    bc = businessCategories[b]
  if u in visitedCategories:
    uc = visitedCategories[u]
  prediction = 0
  if len(bc.intersection(uc)): # User has visited some business of the same category
    prediction = 1
  _ = predictions.write(u + '-' + b + ',' + str(prediction) + '\n')

predictions.close()

##################################################
# Categorization (CSE158 only)                   #
##################################################

reviewsTrain = [r for r in reviewsTrain if 'categoryID' in r]
reviewsValid = [r for r in reviewsValid if 'categoryID' in r]

reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
for r in reviewsTrain:
  reviewsPerUser[r['userID']].append(r)
  reviewsPerItem[r['businessID']].append(r)

catDict = {
  "American Restaurant": 0,
  "Bar": 1,
  "Asian Restaurant": 2,
  "European Restaurant": 3,
  "Italian Restaurant": 4,
  "Fast Food Restaurant": 5,
  "Mexican Restaurant": 6,
  "Seafood Restaurant": 7,
  "Coffee Shop": 8,
  "Sandwich Shop": 9
}

# Trivial predictor
# Counts of category purchases per user
predictionForUser = {}
for u in reviewsPerUser:
  catCount = defaultdict(int)
  for r in reviewsPerUser[u]:
    catCount[r['categoryID']] += 1
  catCounts = [(catCount[c], c) for c in catCount]
  catCounts.sort()
  # Predict the most common category
  predictionForUser[u] = catCounts[-1][1]

validAcc = 0
for r in reviewsValid:
  u = r['userID']
  prediction = 0
  if u in predictionForUser:
    prediction = predictionForUser[u]
  if prediction == r['categoryID']:
    validAcc += 1

validAcc *= 1.0 / len(reviewsValid)

print("Validation accuracy (trivial classifier) = " + str(validAcc))

# Word counts
wordFreq = defaultdict(float)
wordFreqCat = {}
for c in range(10):
  wordFreqCat[c] = defaultdict(float)

punctuation = string.punctuation

for r in reviewsTrain:
  rText = ''.join([c for c in r['reviewText'].lower() if not c in punctuation])
  for w in rText.split():
    wordFreq[w] += 1.0 # Word frequency
    wordFreqCat[r['categoryID']][w] += 1.0 # Per-category frequency

totalWords = sum(wordFreq.values())
for w in wordFreq:
  wordFreq[w] /= totalWords

for c in range(10):
  totalWordsCat = sum(wordFreqCat[c].values())
  for w in wordFreqCat[c]:
    wordFreqCat[c][w] /= totalWordsCat

for c in range(10):
  diffs = []
  for w in wordFreq:
    diffs.append((wordFreqCat[c][w] - wordFreq[w], w))
  diffs.sort()
  diffs.reverse()
  print("10 words with maximum frequency difference for category " + str(c) + " = " + str(diffs[:10]))

# SVM classifiers

topWords = [(wordFreq[w], w) for w in wordFreq]
topWords.sort()
topWords.reverse()

commonWords = [x[1] for x in topWords[:500]]
commonWordsPositions = dict(zip(commonWords, range(len(commonWords))))
commonWordsSet = set(commonWords)

def feature(r):
  feat = [0] * len(commonWords)
  rText = ''.join([c for c in r['reviewText'].lower() if not c in punctuation])
  for w in rText.split():
    if w in commonWordsSet:
      feat[commonWordsPositions[w]] = 1
  return feat

X_train = [feature(r) for r in reviewsTrain if r['categoryID'] < 2]
y_train = [r['categoryID'] for r in reviewsTrain if r['categoryID'] < 2]

X_valid = [feature(r) for r in reviewsValid]
y_valid = [r['categoryID'] for r in reviewsValid]

# Binary classifier for men's/women's clothing

bestAcc = 0
bestCLF = None
for c in 0.01, 0.1, 1, 10, 100:
  clf = svm.LinearSVC(C = c) # Linear SVM is faster
  clf.fit(X_train, y_train)
  predictions = [int(x) for x in clf.predict(X_valid)]
  acc = [(x == y) for (x,y) in zip(predictions, y_valid)]
  acc = sum(acc) * 1.0 / len(acc)
  if acc > bestAcc:
    bestAcc = acc
    bestCLF = clf
  print("C = " + str(c) + ": validation accuracy = " + str(acc))

# Format for Kaggle

predictions = open("predictions_Category.txt", 'w')
predictions.write("userID-reviewHash,category\n")
for l in readGz("test_Category.json.gz"):
  feat = feature(l)
  prediction = int(bestCLF.predict([feat])[0])
  _ = predictions.write(l['userID'] + '-' + l['reviewHash'] + "," + str(prediction) + "\n")

predictions.close()

# Five classifiers for each category

X_train = [feature(r) for r in reviewsTrain]

clfs = {}
for cat in range(10):
  y_trainC = [r['categoryID'] == cat for r in reviewsTrain]
  y_validC = [r['categoryID'] == cat for r in reviewsValid]
  bestAcc = 0
  bestCLF = None
  for c in 0.01, 0.1, 1, 10, 100:
    clf = svm.LinearSVC(C = c)
    clf.fit(X_train, y_trainC)
    predictions = [x for x in clf.predict(X_valid)]
    acc = [(x == y) for (x,y) in zip(predictions, y_validC)]
    acc = sum(acc) * 1.0 / len(acc)
    print("cat = " + str(cat) + ", C = " + str(c) + ": validation accuracy = " + str(acc))
    if acc > bestAcc:
      bestAcc = acc
      bestCLF = clf
  clfs[cat] = bestCLF

confidences = {}
for cat in range(10):
  confidences[cat] = clfs[cat].decision_function(X_valid)

predictions = []
for i in range(len(confidences[0])):
  cs = [(confidences[c][i],c) for c in range(5)]
  cs.sort()
  mostConfidentCategory = cs[-1][1]
  predictions.append(mostConfidentCategory)

validAcc = [(x == y) for (x,y) in zip(predictions, y_valid)]
validAcc = sum(validAcc) * 1.0 / len(validAcc)

print("Multi-SVM accuracy = " + str(validAcc))

# Can do something like the above much more easily just using a multiclass SVM (with the library)

y_train = [r['categoryID'] for r in reviewsTrain]
y_valid = [r['categoryID'] for r in reviewsValid]

clf = svm.LinearSVC(C = 1)
clf.fit(X_train, y_train)

clf.predict(X_valid) # Will be a vector containing numbers from 0-4
