# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:21:15 2018

@author: Shushan
"""

import numpy as np
from skimage import feature



#function for projection profile for 1 column
def ColumnProjectionProfile(column):
    s = np.sum(column)
    return column.shape[0] - s


#two 2D numpy arrays, where the de 0 dimension must match
def ProjectionProfile(img):
    #apply the function to all columns of the array
    result = np.apply_along_axis(ColumnProjectionProfile, 0, img)
    #normalise the results to range [0,1]
    result_normalised = (result - np.min(result)) / (np.max(result) - np.min(result))
    return result_normalised

def ColumnProfile(column, up):
    '''
    Extract the column profile: if 2nd argument is 0 then it is the upper profile
    if it is -1 then it is the lower one.
    '''

    try:
        return np.nonzero(1 - column)[0][up]
    except:
        return np.nan

def WordProfile(img, up):
    up = np.apply_along_axis(ColumnProfile, 0, img, up)
    nans, x= nan_helper(up)
    up[nans]= np.interp(x(nans), x(~nans), up[~nans])
    up_normalised = (up - np.min(up))/ (np.max(up) - np.min(up))
    return up_normalised

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def Transitions(img):
    edges = feature.canny(img, sigma=1) 
    edges.astype(int)
    edges_sum = np.sum(edges, axis = 0)
    transitions_normalised = (edges_sum - np.min(edges_sum))/(np.max(edges_sum) - np.min(edges_sum))
    return transitions_normalised

def ColumnFractionBlack(col):
    return sum(col!=1)/len(col)

def FractionBlack(img):
    fb = np.apply_along_axis(ColumnFractionBlack, 0, img)
    return fb

def CreateFeatures(img):
    '''
    Function takes the numpy matrix of the image and outputs a matrix of features 
    where the rows correspond to image positions and the columns to the individual features
    '''
    f1 = ProjectionProfile(img) #the distribution of ink along one of the two dimensions in a word image
    f2 = WordProfile(img, 0) #upper profile, i.e. highest point where there is ink
    f3 = WordProfile(img, -1) #lower profile, i.e. lowest point where there is ink
    f4 = Transitions(img) #number of black to white and white to black transitions
    f5 = FractionBlack(img) #fraction of ink in each column
    feature_vectors = np.stack((f1, f2, f3, f4, f5))
    return feature_vectors

def CreateAllFeatures(imgs):
    features = []
    for img in imgs:
        features.append(CreateFeatures(img))
    return features