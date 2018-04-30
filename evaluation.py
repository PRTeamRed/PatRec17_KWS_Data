
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

from PIL import Image
from numpy import array

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import operator
import DTWDistance
#*************** Create multi-label data, fit, and predict


def main():
	
	#************** TODO: prepare data *****************
	"""
	load testing_samples
	load training_samples
	What is the structure of these datasets?
	
	"""
	
	
	# generate predictions
	predictions=[]
	k = 1
	for testing_sample in testing_samples:
		# Get the k nearest neighbors through DTW distance algorithm
		kNN_DTW = getkNN_DTW(training_samples, testing_sample], k)
		# Get the label occuring the most among the k nearest neighbors
		result = getResponse(kNN_DTW)
		predictions.append(result)
	# Get the accuracy. Compares test_label with prediction_label
	labels_score = getAccuracy(testing_samples,predictions)
	print('Accuracy: ' + repr(labels_score) + '%')
	
	"""
	*********** TODO **************
	labels_test = labels of the testing set
	n_labels = number of possible labels
	"""
	precision, recall, average_precision = compute_average_precision_score(n_labels, labels_test, labels_score)
	# Plot precision-recall curve
	plot_precision_recall_curve(precision,recall,average_precision)
                                                    


def getkNN_DTW(training_samples, testing_sample, k):
# Returns the k nearest neighbors through DTW distance algorithm

	distance = []
	# compute distance between each training instance and the test instance
	# and store tuple of result and corresponding training instance
	# into array of distances.
	for training_sample in training_samples:
		#dist = euclideanDistanceAlgo(testing_sample, training_sample)
		#****************** TODO: How can I extract the feature vectors..???? ******************************
		dist = DTWDistance.DTWDistance(feature_vector1_testing_sample, feature_vector2_training_sample)
		distances.append((training_sample, dist))
	# sort list of tuples of distances in ascending order 
	# regarding the second item of the tuples, i.e. the distances	
	distances.sort(key=operator.itemgetter(1))
	kNN = []
	for x in range(k):
		kNN.append(distances[x][0])
	# return the k first training instances (the k most similar)
	return kNN											


def getResponse(kNN):
# Returns the label occuring the most among the k nearest neighbors

	# classVotes is a key:value list
	classVotes = {}
	for x in range(len(kNN)):
		# extract the label of training instance 
		response = kNN[x][0]
		# if the label appears in list of classVotes 
		if response in classVotes:
		# add vote at to label key
			classVotes[response] += 1  
		else:
			classVotes[response] = 1
	# sort the classVotes according to the number of votes in a descending order
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
	


def getAccuracy(testing_samples, predictions):
# Returns the accuracy of predictions in %, which is the ratio
# of the total correct predictions out of all predictions made
	correct = 0
	for x in range(len(testing_samples)):
		# if test label is equal to prediction label
		if testing_samples[x][0] is predictions[x]:
			correct += 1
		# return ratio
	return (correct/float(len(testSet)))*100.0


def compute_average_precision_score(n_labels, labels_test, labels_score):
#**************** Compute the average precision score
	# For each class
	precision = dict() # {}
	recall = dict() # {}
	average_precision = dict() # {}
	for i in range(n_classes):
		precision[i], recall[i], _ = precision_recall_curve(labels_test[:, i],labels_score[:, i])
		average_precision[i] = average_precision_score(labels_test[:, i], labels_score[:, i])

	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(labels_test.ravel(),labels_score.ravel())
	average_precision["micro"] = average_precision_score(labels_test, labels_score,
                                                     average="micro")
	print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))
	
	return precision, recall, average_precision


def plot_precision_recall_curve(recall, precision, average_precision):	  
#****************** Plot the Precision-Recall curve

	plt.figure()
	plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
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



if __name__ == "__main__":
    main()

