import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# import some toy data
iris = datasets.load_iris()
data = iris.data
target = iris.target
n_samples, n_features = data.shape
n_classes = np.unique(target).shape[0]

# 60% subset for training 
# 40% subset for testing 
train_proportion = 0.6

# split training and testing sets
perm = np.random.permutation(n_samples)
data = data[perm]
target = target[perm]
split = int(n_samples*train_proportion)
data_train = data[:split]
data_test = data[split:]
target_train = target[:split]
target_test = target[split:]

# classify and calculate scores
classifier = OneVsRestClassifier(svm.SVC(kernel='linear'))
y_score = classifier.fit(data_train,target_train).decision_function(data_test)

# binarize the target classes of test set
y_true = label_binarize(target_test,classes=[0,1,2])

# for each class, compute ROC curve and AUC
fpr,tpr = dict(),dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:,i],y_score[:,i])
    roc_auc[i] = auc(fpr[i],tpr[i])

# for each class, plot ROC curve 
fig = plt.figure()
ax1 = fig.add_subplot()
for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],'o-',\
        label='ROC curve of class {0}(area ={1:0.4f})'.format(i,roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')    
plt.xlim([-0.05,1.0])
plt.ylim([0.0,1.05])



# compute micro ROC curve and AUC
# each document weights equal
micro_true = y_true.ravel()
micro_score = y_score.ravel()
fpr['micro'], tpr['micro'], _ = roc_curve(micro_true,micro_score)
roc_auc['micro'] = auc(fpr['micro'],tpr['micro'])


# compute macro AUC
# each class weights equal
roc_auc['macro2'] = np.average([roc_auc[i] for i in range(n_classes)])

# compute macro ROC curve

# the false_positive_rate samples for differnet classes vary
# get all false_positive_rate samples
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# calculate average true_positive_rates at all false_positive_rates samples,
# using linear interpolation
all_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    all_tpr += interp(all_fpr, fpr[i], tpr[i])

all_tpr = all_tpr/float(n_classes)
roc_auc['macro'] = auc(all_fpr, all_tpr)

plt.plot(fpr['micro'],tpr['micro'],'o-',\
    label='micro-ROC curve (area ={0:0.4f})'.format(roc_auc['micro']))

plt.plot(all_fpr,all_tpr,'o-',\
    label='macro-ROC curve (area ={0:0.4f})'.format(roc_auc['macro']))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for multi-class')
plt.legend(loc="lower right")
plt.show()





