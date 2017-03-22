'''
    AUTHOR Archith Shivanagere
'''

import numpy as np
import matplotlib.pyplot as plt
import tabulate as tb
from datetime import datetime
from functools import partial
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from matplotlib.backends.backend_pdf import PdfPages

vectorCount = {'stop_words': 'english'}
parametersTFIDF = {'sublinear_tf': True}
clfs = { 'Naive Bayes': Pipeline([('vect',CountVectorizer(**vectorCount)),('tfidf',TfidfTransformer(**parametersTFIDF)),('clf',MultinomialNB())]),
         'SVM': Pipeline([('vect',CountVectorizer(**vectorCount)),('tfidf',TfidfTransformer(**parametersTFIDF)),('clf',svm.SVC(kernel = cosine_similarity, probability = True))])
         }
classes = ['comp.graphics', 'comp.sys.mac.hardware', 'rec.motorcycles', 'sci.space', 'talk.politics.mideast']
trainData = fetch_20newsgroups(subset = 'train', remove = ('quotes', 'headers', 'footers'), categories = classes)
testData = fetch_20newsgroups(subset = 'test', remove = ('quotes', 'headers', 'footers'), categories = classes)
count = 7

plt.figure()

falsePositiveSVM = dict()
truePositiveSVM = dict()
aucSVM = dict()
falsePositiveNB = dict()
truePositiveNB = dict()
aucNB = dict()

values = [[] for _ in xrange(count)]

for i in clfs:
    
    prev = datetime.now()
    classifier = clfs[i].fit(trainData.data, trainData.target)
    cur = datetime.now()
    trainingTime = str(cur - prev)
    values[0].append(trainingTime)

    predictions = classifier.predict_proba(testData.data)
    for j, k in enumerate((metrics.accuracy_score, partial(metrics.precision_score, average = 'micro'), partial(metrics.recall_score, average = 'micro'))):
        trainValues = k(trainData.target,classifier.predict(trainData.data))
        testValues = k(testData.target,classifier.predict(testData.data))
        trainStr = str(trainValues)
        testStr = str(testValues)
        values[1 + 2 * j].append(trainStr)
        values[2 + 2 * j].append(testStr)
    for m, n in enumerate(classes):
        if i == 'SVM':
            falsePositiveSVM[n], truePositiveSVM[n], _ = roc_curve(testData.target, predictions[:, m], pos_label = m)
            aucSVM[n] = auc(falsePositiveSVM[n], truePositiveSVM[n])
            plt.plot(falsePositiveSVM[n], truePositiveSVM[n], label='ROC SVM curve of class {0} (area = {1:0.3f})'
			                                    ''.format(n, aucSVM[n]))
        else:
            falsePositiveNB[n], truePositiveNB[n], _ = roc_curve(testData.target, predictions[:, m], pos_label = m)
            aucNB[n] = auc(falsePositiveNB[n], truePositiveNB[n])
            plt.plot(falsePositiveNB[n], truePositiveNB[n], label='ROC Naive Bayes curve of class {0} (area = {1:0.3f})'
			                                    ''.format(n, aucNB[n]))

rows = ["Time", "Accuracy", "Precision", "Recall"]
clfs_names = ["Metric", "Naive Bayes", "SVM with cosine kernel"]
values = [[row_name] + values[row_num]
          for row_num, row_name in enumerate(rows)]
print "\n"
print tb.tabulate(values, clfs_names)
print "\n"

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Plot Naive Bayes and SVM Classifiers')
plt.legend(loc="lower right", prop={'size':5})
plt.savefig("graphTextClassifierROC.pdf")
plt.close()