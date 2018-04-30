
"""
main sources:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
https://www.quora.com/How-do-I-read-image-data-for-CNN-in-python    to load data
Author: Julia
"""
from sklearn.preprocessing import label_binarize

task_path = "task/"  # train.txt, valid.txt, keywords.txt
img_path = "images/"  # .jpg
labels_path = "ground-truth/locations/" # .svg

from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import train_test_split
import numpy as np

#*************** Create multi-label data, fit, and predict

#### Load data
from PIL import Image
from numpy import array
""""
# [270.jpg, 271.jpg, ...]
images_name_jpg = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]
images = []
for inj in image_names_jpg
    img = Image.open(img_path + inj)
    img_arr = array(img)
    images.append(img_arr)
"""
"""
data = load_data()
images = data.images  # X
l = data.labels # y
"""

# Use label_binarize to be multi-label like settings
labels = label_binarize(l, classes=[0, 1, 2])  # Y
n_classes = labels.shape[1]


# Split into training and test
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=.5,
                                                    random_state=random_state)
                                                    
# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

# Run keywords spotter  
"""
kwspotter = KWSpotter(...)
kwspotter.fit(images_train, labels_train)
labels_score = kwspotter.decision_function(images_test)
"""


#**************** Compute the average precision score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict() # {}
recall = dict() # {}
average_precision = dict() # {}
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(labels_test[:, i],
                                                        labels_score[:, i])
    average_precision[i] = average_precision_score(labels_test[:, i], labels_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(labels_test.ravel(),
    labels_score.ravel())
average_precision["micro"] = average_precision_score(labels_test, labels_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


#****************** Plot the Precision-Recall curve

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))
plt.show()  




