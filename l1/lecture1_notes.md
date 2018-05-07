# CS231n - Lecture 1 notes (slides)

## Introduction to Convolutional Neural Network for Visual Recognition

Visual recognition problems:
* Image classification
    
    - Object detection
    
    - Image captioning

    - Action classification

# Lecture 1 notes (video)

[Stanford University CS231n, Spring 2017 - Lecture 1](https://www.youtube.com/watch?v=vT1JzLTH4G4&index=0&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)

## Larry Roberts, 1963 - Block World
- Goal: Able to recognize and reconstruct the shapes

## David Marr - Vision
Input image

Edge image
- Zero crossings, blobs, edges, curves, ends, etc.

2 1/2-D sketch
- Local surface orientation, layers (discontinuities) in depth and surface orientation

3-D model
- 3-D models hierarchically organized in terms of surface and volume primitives

Every object is composed of simple geometric primitives

## David Lowe, 1987
- Constructing lines, edges

## Shi & Malik, 1997 - Normalized Cut
Problem of Object Recognition solved by:

- Object Segmentation: task of taking an image and group the pixels to meaningful areas.

## Viola & Jones, 2001 - Face Detection
- SVM, Boosting (Adaboost algorithm for real-time face detection), Graphical Model, Neural Networks


Feature based object recognition
- SIFT: match an entire object to another object (camera angles, occlusion, viewpoint, lighting, intrinsic variation of the object)

Feature that tend to remain in diagnostic an invariant to changes

Identifying these critical features and match these features to a similar object

Easier task than pattern matching an entire object

## Schmid & Ponce, 2016 - Spatial Pyramid Matching
Take features from different parts of the image in different resolution, put them together in a feature descriptor and do support vector machine

## ImageNet
22K categories and 14M images

Large visual database designed for use in visual object recognition software research


## ImageNet Challenge
AlexNet

GoogLeNet

VGG

MSRA

Increasing computing (GPUs, transistors) with the same classical approach and algorithms tends to work well

Data (quality and labeled datasets)

Semantic segmentation, perception grouping
- Understand for each pixel what does it do and mean

## Johnson & al. - Image Retrieval using Scene Graphs
Describing objects as a graph of semantic related concepts that encompass objects identities, relationships, attributes, actions occuring in the scene