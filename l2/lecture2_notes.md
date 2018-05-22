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
1. `Input`: Collect a dataset of images and labels.
- Our input consists of a set of N images, each labeled with one of K different classes. We refer to this data as the training set.
2. `Learning`: Use Machine Learning to train a classifier.
- Our task is to use the training set to learn what every one of the classes looks like. We refer to this step as training a classifier, or learning a model.
3. `Evaluation`: Evaluate the classifier on new images.
-  In the end, we evaluate the quality of the classifier by asking it to predict labels for a new set of images that it has never seen before. We will then compare the true labels of these images to the ones predicted by the classifier.

```python
def train(images, labels):
    ...
    return model
```
```python
def predict(model, test_images):
    ...
    return test_labels
```

### First Classifier: Nearest Neighbor classifier

Take a new image and find the most similar image in the training data and predict the label of the most similar image

CIFAR10 Dataset
- 10 classes
- 60,000 tiny images that are 32 pixels high and wide
    - 50,000 training images
    - 10,000 testing images

### Distance metric
#### L1 Distance (Manhattan distance)
Sum of pixel-wise absolute value differences

Train: O(1)

Predict: O(N)

We want classifiers that are `fast` at prediction; `slow` for training is ok

## k-Nearest Neighbors
An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its *k* nearest neighbors.
If k = 1, then the object is simply assigned to the class of that single nearest neighbor. Intuitively, higher values of k have a smoothing effect that makes the classifier more resistant to outliers:


- Load CIFAR-10 data into memory and stretch out the images as rows
```python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
```
```python
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```
```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```
- This classifier only achieves 38.6% on CIFAR-10

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

**In practice**, people prefer to avoid cross-validation in favor of having a single validation split, since cross-validation can be computationally expensive. Typical number of folds you can see in practice would be 3-fold, 5-fold or 10-fold cross-validation.

The splits people tend to use is between 50% to 90% for the training data and the rest for the validation set.

https://en.wikipedia.org/wiki/Training,_test,_and_validation_sets

### Validation set
- Don't have direct access to the labels comparatively to the training set. 

- Select best performing approach using the validation data

```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
  
  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

### Test set
- Estimate the accuracy, sensitivity, specificity, F-measure, etc. of the selected approach with the test data.

**Run the test set only once at the very end, it remains a good proxy for measuring the `generalization` of your classifier.**

**We cannot use the test set for the purpose of tweaking hyperparameters. In practice, we would say that you `overfit` to the test set.**

## Q: Test set might not be representative of the data out in the wild.
Partitioning randomly among the entire set of datapoints alleviate this problem in practice. Not by using the earlier data as the training dataet and the newest for the test data if your collecting data overtime.

k-Nearest Neighbors on images `never used`.
- Very slow at test time.
- Distance metrics on pixels are not informative.
- Curse of dimensionality: The number of training examples needed to densely cover the space grows exponentially with the dimension.
- Images are high-dimensional objects (i.e. they often contain many pixels)

## Linear Classification
Parametric Approach/Model

### CIFAR10 Dataset

**Example** :

* Image x: array of 32x32x3 (3072 numbers)
* W: parameters or weights
* b: bias vector

**First component**: `Score function` that maps the raw data to confidence score for each class

f(x,W) -> 10 numbers giving class scores

`f(xi,W,b) = Wxi + b`

We are assuming that the image `xi` has all of its pixels flattened out to a single column vector of shape [D x 1]. The matrix `W` (of size [K x D]), and the vector `b` (of size [K x 1]) are the parameters of the function.

D = 32x32x3 = 3072 pixels

K = 10 distinct classes

10x1 = 10x3072 * 3072x1 + 10x1

An advantage of this approach is that the training data is used to learn the parameters `W,b`, but once the learning is complete we can discard the entire training set and only keep the learned parameters. That is because a new test image can be simply forwarded through the function and classified based on the computed scores.

The problem is that the linear classifier is only learning one template for each class. If there are variations in how the class might appear, it tries to average out all those different variations and only use one single template to recognize each of those categories.

Another problem is the linear decision boundaries of the linear classifier that tries to draw linear separation between one category and the rest of the categories in a high dimensional space.

Functional form corresponding to a linear classifier: `Template matching` and learning a single template for each category of the data.

### Bias trick

A commonly used trick is to combine the two sets of parameters into a single matrix that holds both of them by extending the vector xi with one additional dimension that always holds the constant 1 - a default bias dimension. With the extra dimension, the new score function will simplify to a single matrix multiply:

`f(xi,W) = Wxi`

With our CIFAR-10 example, `xi` is now [3073 x 1] instead of [3072 x 1] - (with the extra dimension holding the constant 1), and `W` is now [10 x 3073] instead of [10 x 3072]. The extra column that `W` now corresponds to the bias `b`.

![](resources/wb.jpeg)

### Image data preprocessing

In the examples above we used the raw pixel values (which range from [0…255]). In Machine Learning, it is a very common practice to always perform `normalization` of your input features (in the case of images, every pixel is thought of as a feature). 

In particular, it is important to `center your data` by subtracting the mean from every feature. In the case of images, this corresponds to computing a mean image across the training images and subtracting it from every image to get images where the pixels range from approximately [-127 … 127]. 

Further common preprocessing is to scale each input feature so that its values range from [-1, 1]. Of these, zero mean centering is arguably more important.


## Summary

- We introduced the problem of `Image Classification`, in which we are given a set of images that are all labeled with a single category. We are then asked to predict these categories for a novel set of test images and measure the accuracy of the predictions.

- We introduced a simple classifier called the `Nearest Neighbor classifier`. We saw that there are multiple hyper-parameters (such as value of k, or the type of distance used to compare examples) that are associated with this classifier and that there was no obvious way of choosing them.

- We saw that the correct way to set these hyperparameters is to split your training data into two: a training set and a fake test set, which we call `validation set`. We try different hyperparameter values and keep the values that lead to the best performance on the validation set.

- If the lack of training data is a concern, we discussed a procedure called `cross-validation`, which can help reduce noise in estimating which hyperparameters work best.

- Once the best hyperparameters are found, we fix them and perform a `single evaluation` on the actual test set.

- We saw that Nearest Neighbor can get us about 40% accuracy on CIFAR-10. It is simple to implement but requires us to store the entire training set and it is expensive to evaluate on a test image.

- Finally, we saw that the use of L1 or L2 distances on raw pixel values is not adequate since the distances correlate more strongly with backgrounds and color distributions of images than with their semantic content.

## References

http://cs231n.github.io/classification/