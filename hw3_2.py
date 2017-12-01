import random
from collections import defaultdict
import gzip
import string
from sklearn.svm import LinearSVC

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

# Creating a initial training and validation set from train.json.gz
data = []
train = []
validate = []
user_list = set()
biz_list = set()
count_i = 0
for l in readGz("train.json.gz"):
    if l.get('categoryID') >= 0:
        user, biz, cat = l['userID'], l['businessID'], int(l['categoryID'])
        data.append(l)
        user_list.add(user)
        biz_list.add(biz)
        count_i +=1

# Creating the training and validation through random subsampling
train = data[:(len(data)/2)]
validate = data[(len(data)/2)+1:]
user_list = list(user_list)
biz_list = list(biz_list)


### Category prediction baseline: Just consider some of the most common words from each category

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

revcatDict = {
  0: "American Restaurant",
  1: "Bar",
  2: "Asian Restaurant",
  3: "European Restaurant",
  4: "Italian Restaurant",
  5: "Fast Food Restaurant",
  6: "Mexican Restaurant",
  7: "Seafood Restaurant",
  8: "Coffee Shop",
  9: "Sandwich Shop"
}

# Populating the user categories and the most popular categories based
# on training data
userCat = defaultdict(list)
catPop = defaultdict(int)
for l in train:
    user, cat = l['userID'], l['categoryID']
    cat_list = userCat[user]
    if len(cat_list) < 1:
        cat_list = [0 for a in range(0, 10)]
        cat_list[cat] += 1
    else:
        cat_list[cat] += 1
    userCat[user] = cat_list
    catPop[cat] += 1

print " Category popularities and user calculated."

# Gets the users favourite category and returns the most popl category if
def userFavCat(user, cat_popl):
    cat_list = userCat[user]
    if cat_list == []:
        return 0
    max_count = 0
    indexes = []
    for count in cat_list:
        if max_count > count:
            continue;
        elif max_count == count:
            indexes.append(cat_list.index(count))
        else:
            max_count = count
            while len(indexes) > 0:
                indexes.pop()
            indexes.append(cat_list.index(count))
    popl = [cat_popl[idx] for idx in indexes]
    retval = indexes[popl.index(max(popl))]
    return retval
'''
predictions = open("predictions_Category6.txt", 'w')
predictions.write("userID-reviewHash,category\n")
for l in validate:
  cat = userFavCat(l['userID'], catPop)
  predictions.write(l['userID'] + '-' + l['reviewHash'] + "," + str(cat) + "\n")

predictions.close()

idx = 0
correct = 0
for line in open("predictions_Category6.txt", 'r'):
    if line.startswith("userID"):
        continue;
    user, review = line.strip().split('-')
    review_hash, cat = review.split(',')
    l = validate[idx]
    idx += 1
    if int(cat) == l['categoryID']:
        correct += 1

print "Accuracy of the category predictor = ", correct*1.0/len(validate)'''

### Just take the most popular words...

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['reviewText'].lower() if not c in punctuation])
  words = r.split()
  phrases = [(words[i],words[i+1]) for i in range(len(words)-1)]
  for w in words:
    wordCount[w] += 1
  for word1, word2 in phrases:
    phraseCount[(word1, word2)] += 1

counts = [(wordCount[w], w) for w in wordCount]
phrase_counts = [(phraseCount[(w1,w2)], (w1,w2)) for (w1,w2) in phraseCount]
counts.sort()
counts.reverse()
phrase_counts.sort()
phrase_counts.reverse()

words = [x[1] for x in counts[:500]]
phrases = [x[1] for x in phrase_counts[:500]]

def cat_data(cat):
    return [line for line in data if (line['categoryID'] == cat)]

pop_words = defaultdict(list)
for i in range(0,10):
    wordCatCount = defaultdict(int)
    phraseCatCount = defaultdict(int)
    punctuation = set(string.punctuation)
    bar_data = cat_data(i)
    #print len(bar_data)
    for d in bar_data:
      r = ''.join([c for c in d['reviewText'].lower() if not c in punctuation])
      words_in_cat = r.split() 
      for w in r.split():
        wordCatCount[w] += 1

    counts = [(wordCatCount[w], w) for w in wordCatCount]
    counts.sort()
    counts.reverse()
    words_2 = [x[1] for x in counts[:500]]
    pop_words[i] = words_2
    print "Category", revcatDict[i]
    print "Popular words", [wd for wd in words_2 if wd not in words][:10]



def label_dataset(datum, i):
    if datum['categoryID'] == i:
        return 1
    else:
        return 0

def feature(datum, wordId, cat_words):
  feat = [0]*len(cat_words)
  r = ''.join([c for c in datum['reviewText'].lower() if not c in punctuation])
  for w in r.split():
    if w in cat_words:
      feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat

def populate_features(i):
  cat_words = pop_words[i]
  y_train = [label_dataset(d, i) for d in train]
  wordId = dict(zip(cat_words, range(len(cat_words))))
  wordSet = set(cat_words)
  X_train = [feature(datum, wordId, cat_words) for datum in train]
  X_validate = [feature(datum, wordId, cat_words) for datum in validate]

  return X_train, y_train, X_validate, wordId, cat_words

def run_predictions():
  best_c_list = []
  for j in range(0,10):
    X_train, y_train, X_validate, wordId, cat_words = populate_features(j)
    c = 0.01
    SVM_predictions = []
    while c <= 100:
      SVM_predictions.append((predictSVM(c, X_train, y_train, X_validate), c))
      c *= 10

    if (j == 1):
      print "Bar SVM models = ", SVM_predictions
    SVM_predictions = sorted(SVM_predictions, key=lambda x: x[0])
    SVM_predictions.reverse()

    best_c = SVM_predictions[0]
    best_c_list.append((j,best_c))
    print "For category", revcatDict[j], "Best (acc, c) = ", best_c

  return best_c_list

def predictSVM(c_val, X_tr, y_tr, X_valid):
  clf = LinearSVC(C=c_val)
  clf.fit(X_tr, y_tr)
  coef = clf.coef_
  intercept = clf.intercept_
  predictions = clf.predict(X_valid)
  dec = clf.score(X_valid, y_tr)
  return dec



def createBestSVMs(best_models):
  svms = []
  for j in range(0,10):
    X_tr, y_tr, X_val, wordId, cat_words = populate_features(j)
    cat, model = best_models[j]
    acc, c_val = model
    clf = LinearSVC(C=c_val)
    clf.fit(X_tr, y_tr)
    svms.append(clf)
  return svms

best_models = run_predictions()
svms = createBestSVMs(best_models)

def predictCat(l):
  dec_func = []
  for cat in range(0,10):
    cat_words = pop_words[cat]
    wordId = dict(zip(cat_words, range(len(cat_words))))
    clf = svms[cat]
    X = [feature(l, wordId, cat_words)]
    dec_func.append(clf.decision_function(X))
  cat = dec_func.index(max(dec_func))
  if cat == None:
    cat = catDict['American Restaurant'] # If there's no evidence, just choose the most common category in the dataset
  return cat

predictions = open("predictions_Category8.txt", 'w')
predictions.write("userID-reviewHash,category\n")
for l in validate:
  cat = catDict['American Restaurant'] # If there's no evidence, just choose the most common category in the dataset
  cat = predictCat(l)
  predictions.write(l['userID'] + '-' + l['reviewHash'] + "," + str(cat) + "\n")

predictions.close()

idx = 0
correct = 0
for line in open("predictions_Category8.txt", 'r'):
    if line.startswith("userID"):
        continue;
    user, review = line.strip().split('-')
    review_hash, cat = review.split(',')
    l = validate[idx]
    idx += 1
    if int(cat) == l['categoryID']:
        correct += 1

print "Accuracy of the category predictor on validation set = ", correct*1.0/len(validate)