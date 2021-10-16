# caloml-atlas

Machine Learning toolkit for calorimeter topo-cluster classification and regression using simulated data from the ATLAS experiment. 

Feel free to contact joakim.olsson[at]cern.ch if you'd like to contribute! 

Machine Learning is awesome :)  

## Image pre-processing

Images are created from ESD (Event Summary Data) files using the [MLTree](https://github.com/jmrolsson/MLTree) Athena package, which generates a root TTree that contains the images as well as some other info. Six images are saved for each cluster, corresponding to the barrels layers of the EM (EMB1, EMB2, EMB3) and HAD (TileBar0, TileBar2, TileBar3) calorimeter. Normalized cell energies are used as pixel values. The image size is 0.4x0.4 in eta-phi space. 

The outputs from [MLTree](https://github.com/jmrolsson/MLTree) can be converted into numpy arrays with [mltree2array.py](util/mltree2array.py)

## Topo-cluster classification

### The task

Train a classifier to determine which type of particle generated the parton showers in the cluster (e.g. electrons vs. charged pions or charged pions vs. neutral pions).

### Implementation

The following models are implemented:

1. Simple fully-connected Neural Network (flattening the images and only using the 512 pixels in the EMB1 layer).
2. Convolutional Neural Networks using only one layer (preserving the shape of the 2D images).
3. A network with multiple images as inputs, and one output (first couple of ConvNets are trained separately, then flattened and concatenated).

Everything is in the [TopoClusterClassifier.ipynb](classifier/TopoClusterClassifier.ipynb) notebook, so it is easy to modify and play around with! 

TODO
- Implement a network of concatenated ConvNets taking all calorimeter layer images into account.
- Also compare the performance with other ML algorithms; logistic regression, SVD, Naive Bias, Gaussians, etc.

## Energy regression

Coming soon...

## Point Cloud Approach
This is a new approach which uses a set of points in space which represent calorimeter hits or tracks. In the case of deep sets or otherwise, these points have some number of attributes. For the deep set energy regression notebooks this set of attributes is [Cell Energy, Eta, Phi, rPerp, track flag]. An additional calorimeter sample layer number is added at the end for bookkeeping purposes and is not used in the training. The numbers are 1,2,3,12,13,14 for EMB1 -> TileBar2 respectively.

#### Instructions to run scripts to create data:
