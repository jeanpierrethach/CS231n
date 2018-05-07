# CS231n - Lecture 2 notes

[Stanford University CS231n, Spring 2017 - Lecture 2](https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=1)


## Image Classification pipeline

Google Cloud

http://cs231n.github.io/gce-tutorial/


### Problem: Semantic Gap

Image : grid of numbers between [0, 255]

e.g. 800 x 600 x 3 
* (3 channels RGB)

### Challenges
* Viewpoint variation: A single instance of an object can be oriented in many ways with respect to the camera.
* Scale variation. Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image).
* Illumination conditions. The effects of illumination are drastic on the pixel level.
* Deformation: Many objects of interest are not rigid bodies and can be deformed in extreme ways.
* Occlusion: The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels) could be visible.
* Background Clutter: The objects of interest may blend into their environment, making them hard to identify.
* Intraclass Variation: The classes of interest can often be relatively broad, such as chair. There are many different types of these objects, each with their own appearance, shape, size, color, etc.

### Machine Learning: Data-Driven Approach
1. Collect a dataset of images and labels
2. Use Machine Learning to train a classifier
3. Evaluate the classifier on new images

```
def train(images, labels):
    ...
    return model
```
```
def predict(model, test_images):
    ...
    return test_labels
```

### First Classifier: Nearest Neighbor classifier

Take a new image and find the most similar image in the training data and predict the label of the most similar image

CIFAR10 Dataset

### Distance metric
#### L1 Distance (Manhattan distance)
Sum of pixel-wise absolute value differences

Train: O(1)
Predict: O(N)

We want classifiers that are `fast` at predication; `slow` for training is ok

## k-Nearest Neighbors
An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its *k* nearest neighbors.
If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

#### L2 Distance (Euclidean distance)
Square root of the sum of the squares

L1 distance depends on the choice of coordinates system, if we were to rotate the coordinate frame, that would change the L1 distance between the points.
As for L2 distance, it is the same thing no matter what the coordinate system is.

### Hyperparameters
Choices about the algorithm that we set rather than learn

#### Idea #1: Choosing hyperparameters that work best on the training data. 

`Terrible` since it will always work perfectly on training data.

#### Idea #2: Split data into `train` and `test`, choose hyperparameters that work best on test data. 

`Terrible` because we have no idea on how the algorithm will perform on new data (may work quite well on this testing set, will no longer be representative of our performance on new unseen data).

#### Idea #3: Split data into `train, validation,` and `test`; choose hyperparameters on validation and evaluate on test.

`Better`

#### Idea #4: `Cross-Validation`: Split data into `folds`, try each fold as validation and average the results.

Cycle choosing which fold will be the validation set from the folds.

Useful for small datasets, but not used too frequently in deep learning.

https://en.wikipedia.org/wiki/Training,_test,_and_validation_sets

### Validation set
Don't have direct access to the labels comparatively to the training set
Select best performing approach using the validation data

### Test set
Estimate the accuracy, sensitivity, specificity, F-measure, etc. of the selected approach with the test data.

`Run the test set only once at the very end`

## Q: Test set might not be representative of the data out in the wild.
Partitioning randomly among the entire set of datapoints alleviate this problem in practice. Not by using the earlier data as the training dataet and the newest for the test data if your collecting data overtime.

k-Nearest Neighbors on images `never used`.
- Very slow at test time.
- Distance metrics on pixels are not informative.
- Curse of dimensionality: The number of training examples needed to densely cover the space grows exponentially with the dimension.

## Linear Classification
Parametric Approach/Model

### CIFAR10 Dataset

* Image x: array of 32x32x3 (3072 numbers)
* W: parameters or weights
* b: bias term

f(x,W) -> 10 numbers giving class scores

`f(x,W) = Wx + b`

10x1 = 10x3072 * 3072x1 + 10x1

The problem is that the linear classifier is only learning one template for each class. If there are variations in how the class might appear, it tries to average out all those different variations and only use one single template to recognize each of those categories.

Another problem is the linear decision boundaries of the linear classifier that tries to draw linear separation between one category and the rest of the categories in a high dimensional space.

Functional form corresponding to a linear classifier: `Template matching` and learning a single template for each category of the data.