# Introduction to Deep Learning for computer vision with Python Deep learning has achieved significant breakthroughs in a wide range of
applications, including image and speech recognition, natural...

::::### **Introduction to Deep Learning for computer vision with Python** 

#### Deep learning has achieved significant breakthroughs in a wide range of applications, including image and speech recognition, natural language processing, autonomous vehicles, and robotics. It has revolutionized the field of artificial intelligence and is one of the key driving forces behind recent advancements in machine learning and data science.
Deep learning is a subfield of machine learning that uses artificial
neural networks to learn and make predictions or decisions from data.
These neural networks are composed of layers of interconnected nodes,
called neurons, that process information and extract features from input
data. By training these networks on large datasets, deep learning
algorithms can automatically learn to recognize patterns and make
accurate predictions or classifications on new data.


<figcaption>Photo by <a
href="https://unsplash.com/@theshubhamdhage?utm_source=medium&amp;utm_medium=referral"
class="markup--anchor markup--figure-anchor"
data-href="https://unsplash.com/@theshubhamdhage?utm_source=medium&amp;utm_medium=referral"
rel="photo-creator noopener" target="_blank">Shubham Dhage</a> on <a
href="https://unsplash.com?utm_source=medium&amp;utm_medium=referral"
class="markup--anchor markup--figure-anchor"
data-href="https://unsplash.com?utm_source=medium&amp;utm_medium=referral"
rel="photo-source noopener" target="_blank">Unsplash</a></figcaption>


Python is one of the most popular programming languages used in deep
learning, and it has many powerful libraries that make it easy to build
and train deep neural networks. Some of the popular deep learning
libraries in Python include TensorFlow, PyTorch, Keras, and Caffe.

In deep learning, we use artificial neural networks to learn from large
datasets and make predictions on new data. These networks are inspired
by the structure and function of the human brain, and they consist of
interconnected layers of neurons that process information and learn to
recognize patterns in the data.

Deep learning has revolutionized many fields, including computer vision,
where it has enabled us to build highly accurate image recognition
systems, object detection, segmentation and tracking. It has also
revolutionized natural language processing, where it has enabled us to
build highly accurate language translation systems, text analysis, and
chatbots.

Deep learning has also enabled us to build highly accurate predictive
models for financial forecasting, customer churn prediction, and many
other applications.

In this tutorial, we will explore the basics of deep learning in Python,
including how to build and train artificial neural networks using
popular libraries such as TensorFlow and Keras. We will also cover
various applications of deep learning, including image classification,
object detection, segmentation, and more.

Whether you're a beginner or an intermediate user, this tutorial will
provide you with the foundational knowledge and practical skills needed
to start building and training deep neural networks in Python.

#### **How Artificial Neural Network is similar to Human Brain**
Artificial neural networks (ANNs) are designed to mimic the structure
and function of the human brain, hence they are similar in many ways.
Here are some similarities between artificial neural networks and human
neural networks:

- **Both have neurons:** ANNs consist of artificial neurons, which are
  the building blocks of the network. Similarly, the human brain
  consists of biological neurons.
- **Both have synapses:** ANNs use synapses to connect neurons, and so
  does the human brain.
- **Both learn from experience:** ANNs are trained on a dataset to
  learn patterns and make predictions, similarly, humans learn from
  experiences and use that knowledge to make decisions.
- **Both can adapt:** ANNs can adapt to new inputs and learn from them,
  similarly, the human brain can adapt to new situations and learn from
  them.
- **Both can generalize:** ANNs can generalize their knowledge to new
  inputs that they have not seen before, similarly, humans can
  generalize their knowledge to new situations.

However, it is important to note that the current artificial neural
networks are far less complex and efficient than the human brain.

For this tutorial, I assume that you are familiar with TensorFlow,
PyTorch, Keras, and Caffe.

#### **Difference between Deep Learning and Deep Neural Network**
Deep learning is a subfield of machine learning that involves training
artificial neural networks to learn from data and make predictions or
decisions based on that data. Deep learning models typically have many
layers, allowing them to learn complex representations of the data.

A deep neural network, on the other hand, is a type of neural network
that has multiple hidden layers. These hidden layers allow the network
to learn complex features of the input data and make more accurate
predictions.

While deep learning refers to the broader field of machine learning that
includes the use of deep neural networks, a deep neural network is a
specific type of neural network architecture that is used in deep
learning.

To find the right solution, it is important to identify the underlying
problem and assess whether deep learning is a feasible approach to solve
it. Once the feasibility is established, relevant data corresponding to
the problem needs to be collected and prepared accordingly. An
appropriate deep

learning algorithm needs to be selected for the task at hand. The chosen
algorithm should be used to train the dataset. Finally, the dataset
should be tested to evaluate the performance of the model.

#### **Real Life Examples**
How to recognize square from other shapes?

1.  [Check the four lines.]
2.  [Is it a closed figure?]
3.  [Are the sides perpendicular from each other?]
4.  [Are all sides equal length?]

Deep Learning is a complex task of identifying the shape and broken down
into simpler tasks at a larger side.

Recognizing an Animal! (Is it a Cat or Dog?) Defining facial features
which are important for classification and system will then identify
this automatically.

#### **Limitations of Deep Learning**
Although deep learning has been shown to be extremely effective in
solving many problems, it still has some limitations that should be
considered. Here are a few:

- **Data dependency:** Deep learning models require a lot of data to be
  trained effectively. Without sufficient data, the models may overfit,
  resulting in poor performance on new, unseen data.
- **Computationally intensive:** Deep learning algorithms can be
  computationally expensive, requiring specialized hardware such as GPUs
  or TPUs to train and run effectively.
- **Interpretability:** Deep learning models are often seen as "black
  boxes" that are difficult to interpret, making it hard to understand
  how the model arrived at a particular decision.
- **Limited transferability:** Deep learning models that are trained
  for a specific task may not be easily transferable to other tasks,
  requiring new models to be developed for each new problem.
- **Lack of common sense:** Deep learning models lack the common sense
  and reasoning abilities of humans, making them prone to making
  mistakes or coming to incorrect conclusions in situations where common
  sense is required.

#### **Advantages of using Deep Learning**
- **High accuracy:** Deep Learning algorithms can achieve high levels
  of accuracy in tasks such as image recognition, speech recognition,
  and natural language processing.
- **Ability to learn features automatically:** Deep Learning algorithms
  can automatically learn useful features from raw data, which can be
  useful in situations where hand-engineered features are difficult to
  design.
- **Scalability:** Deep Learning algorithms can be scaled to large
  datasets and complex problems by using powerful computing resources
  such as GPUs and distributed computing systems.
- **Flexibility:** Deep Learning algorithms can be applied to a wide
  range of problems and can learn from a variety of data types such as
  images, text, and audio.

#### **Cons of using Deep Learning**
- **Large amounts of data required:** Deep Learning algorithms require
  large amounts of labeled data to train, which can be difficult to
  obtain in some domains.
- **Computationally intensive:** Training Deep Learning models can be
  computationally intensive and requires high-performance computing
  resources, such as GPUs or cloud computing services.
- **Black-box models:** Deep Learning models can be difficult to
  interpret, making it challenging to understand why a particular
  decision was made.
- **Overfitting:** Deep Learning models can be prone to overfitting the
  training data, leading to poor generalization performance on unseen
  data.

#### Using OpenCV for computer vision
OpenCV provides a simple way to use pre-trained deep learning models
with OpenCV in Python. OpenCV allows us to perform the following tasks
with ease:

- Load a pre-trained model from disk.
- Pre-process an input image for the model.
- Pass the pre-processed image through the model to obtain output
  classifications.

We cannot *train* deep learning models using OpenCV. However, we can use
this API to apply models that were trained using dedicated deep learning
libraries or tools to efficiently use them directly in OpenCV scripts.

In this demonstration, I will explain the basic process of using a
pre-trained deep learning network that has been trained on the ImageNet
dataset and applying it to input images.

Deep Learning is a rapidly growing area of Machine Learning that is
crucial to explore if you are working in computer vision or image
processing. OpenCV 3.3 has made it possible to use pre trained deep
learning networks from popular frameworks like GoogleLeNet, AlexNet,
SqueezeNet, VGGNet, and ResNet. This means that you do not need to spend
hours training the network as it is already pre-trained, and you can use
the output to make decisions in your application. It is important to
note that OpenCV is not a tool for training networks but is a great tool
for using pre-trained networks within the OpenCV ecosystem. The Deep
Learning module in OpenCV is compatible with various popular network
architectures, and the release notes for this module are available on
the OpenCV repository page.

#### **OpenCV deep learning functions and frameworks**
OpenCV 3.3 supports the Caffe, TensorFlow, and Torch/PyTorch frameworks.

In OpenCV 3.3, we can load images from disk using the functions
**cv2.dnn.blobFromImage** and **cv2.dnn.blobFromImages**. We can import
pre-trained models from different frameworks via the "create" methods
such as **cv2.dnn.createCaffeImporter**,
**cv2.dnn.createTensorFlowImporter**, and
**cv2.dnn.createTorchImporter**. However, it is simpler to use the
"read" methods and load a serialized model directly from disk using
**cv2.dnn.readNetFromCaffe, cv2.dnn.readNetFromTensorFlow,
cv2.dnn.readNetFromTorch**, and **cv2.dnn.readhTorchBlob**. Once we have
loaded a model, we can use the .forward method to propagate the image
forward through the model and obtain the classification result.

#### **Classifying images using deep learning and OpenCV**
Let's build a Python script that can classify input images using OpenCV
and the GoogLeNet model pre-trained on ImageNet, using the Caffe
framework. The GoogLeNet architecture, also known as "Inception". OpenCV
3.3 also supports other architectures such as AlexNet, ResNet, and
SqueezeNet. For now, we will focus on how to load a pre-trained Caffe
model and use it to classify an image in OpenCV. Let's start by adding
the necessary libraries or packages in our code:

Important links to download the necessary files related to pre-trained
models and keywords:

[https://github.com/AleDel/deepdreamer](https://github.com/AleDel/deepdreamer)

touchdesigner/blob/master/models/bvlc_googlenet.caffemodel

[https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt](https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt)

```python
# import the necessary packages
import numpy as np
import argparse
import time
import cv2
from google.colab import drive
from matplotlib import pyplot as plt
```

The first part of the code is used to mount Google Drive, which allows
you to access files saved in your Google Drive account.

```python
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
```

The code below constructs an argument parser that allows you to input
the paths to the input image, pre-trained model, deploy prototxt file,
and ImageNet labels file.

```python
from google.colab import drive
# Mount Google Drive
drive.mount('/content/drive')
# Construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", " - image", required=True, help="path to input image") 
parser.add_argument("-p", " - prototxt", required=True, help="path to Caffe 'deploy' prototxt") 
parser.add_argument("-m", " - model", required=True, help="path to Caffe pre-trained model") 
parser.add_argument("-l", " - labels", required=True, help="path to ImageNet labels (i.e., syn
args = vars(parser.parse_args([" - image", "/content/drive/MyDrive/Colab Notebooks/Images/dog1", " - prototxt", "/content/drive/MyDrive/Colab Notebooks/models/d
" - model", "/content/drive/MyDrive/Colab Notebooks/models/bvlc_
" - labels", "/content/drive/MyDrive/Colab Notebooks/models/syn
```

In the vars() function call, we are passing in the file paths as a list
of strings. This is equivalent to passing them in as command line
arguments.

Now that we've established our arguments, we parse them and store them
in a variable, args , for easy access later.

Let's reads in the input image and ImageNet labels file from disk. The
image is then pre-processed by converting it into a blob using
cv2.dnn.blobFromImage() function.

```python
# load the input image from disk
image = cv2.imread(args["image"])
# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
```

Taking note of the comment in the block below, we use
**cv2.dnn.blobFromImage** to perform mean subtraction to normalize the
input image which results in a known blob shape.

```python
# our CNN requires fixed spatial dimensions for our input image(s)
# so we need to ensure it is resized to 224x224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input; # after executing this command our "blob" now has the shape:
# (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
# Since we've opted to use Caffe, we utilize cv2.dnn.readNetFromCaffe 
# to load our Caffe model definition prototxt and pre-trained model from disk

# load our serialized model from disk
print("[INFO] loading model…")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# output: [INFO] loading model…
# set the blob as input to the network and perform a forward-pass to # obtain our output classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start)) 
# output: [INFO] classification took 0.17889 seconds
```

It is important to note at this step that we aren't training a
CNN --- rather, we are making use of a pre-trained network. Therefore we
are just passing the blob through the network (i.e., forward
propagation) to obtain the result (no back-propagation).

First, we specify blob as our input. Second, we make a start timestamp,
followed by passing our input image through the network and storing the
predictions. Finally, we set an end timestamp, so we can calculate the
difference and print the elapsed time.

``` 
# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
idxs = np.argsort(preds[0])[::-1][:5]
```

Using NumPy, we can easily sort and extract the top five predictions.

```python
from matplotlib import pyplot as plt
# loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
# draw the top prediction on the input image
  if i == 0:
    text = "Label: {}, {:.2f}%".format(classes[idx],
    preds[0][idx] * 100)
    cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 0, 255), 2)

# display the predicted label + associated probability to the
# console
print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
classes[idx], preds[0][idx]))
# display the output image
plt.imshow(image)
plt.show()
# output: [INFO] 1. label: Pembroke, probability: 0.5429
# output: [INFO] 2. label: Cardigan, probability: 0.43598
# output: [INFO] 3. label: dingo, probability: 0.0062387
# output: [INFO] 4. label: Eskimo dog, probability: 0.0042355
# output: [INFO] 5. label: malamute, probability: 0.0011247
```


Let's try the same for another input image as below:

For this purpose, just need to change the filename of image in parser.
i.e. 'waterfall1.jpeg' in this case

```python
# Mount Google Drive
drive.mount('/content/drive')
input_image = "waterfall1.jpeg"
# Construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", " - image", required=True, help="path to input image") parser.add_argument("-p", " - prototxt", required=True, help="path to Caffe 'deploy' prototxt parser.add_argument("-m", " - model", required=True, help="path to Caffe pre-trained model") parser.add_argument("-l", " - labels", required=True, help="path to ImageNet labels (i.e., syn

# load the input image from disk

image = cv2.imread(args["image"])

# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# load our serialized model from disk
print("[INFO] loading model…")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# set the blob as input to the network and perform a forward-pass to
# obtain our output classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
idxs = np.argsort(preds[0])[::-1][:5]

# loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
# draw the top prediction on the input image
  if i == 0:
    text = "Label: {}, {:.2f}%".format(classes[idx],
    preds[0][idx] * 100)
    cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 0, 255), 2)
# display the predicted label + associated probability to the
# console
    print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
    classes[idx], preds[0][idx]))

# display the output image
plt.imshow(image)
plt.show()

[INFO] loading model…
[INFO] classification took 0.15972 seconds
[INFO] 1. label: valley, probability: 0.89233
[INFO] 2. label: cliff, probability: 0.056438
[INFO] 3. label: geyser, probability: 0.010541
[INFO] 4. label: promontory, probability: 0.0098734
[INFO] 5. label: dam, probability: 0.0072059
```


Reference:
[https://pyimagesearch.com/2017/08/21/deep-learning-with-opencv/](https://pyimagesearch.com/2017/08/21/deep-learning-with-opencv/) explained the code in a very simple way
and it was easy to understand. Thanks to pyimage search.

Now that you have completed this project, check out a more complicated
version using image segmentation that I made.

[**Image Segmentation for Computer Vision with Python and CV2**\
*Image Segmentation is the process of dividing an image into multiple
regions or segments, each of which corresponds
to...*medium.com](https://medium.com/@kylejones_47003/image-segmentation-for-computer-vision-with-python-and-cv2-a07a0f70b79d "https://medium.com/@kylejones_47003/image-segmentation-for-computer-vision-with-python-and-cv2-a07a0f70b79d")[](https://medium.com/@kylejones_47003/image-segmentation-for-computer-vision-with-python-and-cv2-a07a0f70b79d)
### Related Stories
- [[Intro to Computer Vision with OpenCV in
  Python](https://medium.com/@kylejones_47003/intro-to-computer-vision-with-opencv-in-python-628eb9fca2db)]
- [[Image Segmentation for Computer Vision with Python and
  CV2](https://medium.com/@kylejones_47003/image-segmentation-for-computer-vision-with-python-and-cv2-a07a0f70b79d)]
- [[Introduction to Apache Spark and
  PySpark](https://medium.com/@kylejones_47003/introduction-to-apache-spark-and-pyspark-3ae3fb4757fa)]
::::::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[October 24, 2023](https://medium.com/p/85f94acc3a6b).

[Canonical
link](https://medium.com/@kyle-t-jones/introduction-to-deep-learning-for-computer-vision-with-python-85f94acc3a6b)

Exported from [Medium](https://medium.com) on November 10, 2025.
