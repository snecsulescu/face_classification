# Test

I addressed Problem 1 and Problem 2 using supervised techniques and Problem 3 using an unsupervised approach.
The output of Problem 1 and 2 will be a machine learning model that takes as input an image and outputs 
the name of the person, or the tag 'unknown'. The output of Problem 3 will be a set of clusters.

## Preparing the data
First I split the data in two sets:
 * a set that will be used in the development of the models
 * a set that will be used in the unittests for checking the accuracy of all the models. 
 This dataset makes around 10% of the original data set  and contains image of people 
 known to the model, as well as unknown people.  

## Problem 1 & 2
I have seen two solutions for these two problems:
* a multi-class classification using the embeddings of each image.
* a binary classification using the distance between the embeddings of two images.

Therefore I implemented two approaches. The first solution is implemented using a KNN model and 
the second solution is implemented using a Sequential model.

Each solution has its pros and cons. 
KNN creates a model that is based only on the observed classes, which makes it the perfect solution 
for Problem 1. However, when a new images is given to the model, it will still be classified as 
one of the people in the training set. To solve this pitfall, when the KNN model returns a class, 
I compute the similarity between the embedding of the new image, and the vectors in the training set.
If the mean of the similarity is lower than a threshold (I set it as 0.9), then I consider that the new image 
belongs to the same class, if not, it belongs to an unknown class.

The sequential model, computes the vectorial distance between a new image and all the images from the training set.
The model verifies if each distance corresponds to two similar images or not. 
Then, I associate a class the new image, the class that has more similar images, normalized by the class length. 
Therefore, this approach can return the 'unknown' label, which makes it a good solution for 
Problem 2. For Problem 1, when the model returns 'unknown', the system labels the same image 
using the KNN model.       

### Results

#### Problem 1

* Accuracy for KNN: 0.9971

* Accuracy for Sequential + KNN: 1.0000

The results are good for both approaches, but there is a huge difference in running time:

* Time for KNN: 3.437656875050396 ms

* Time for Sequential + KNN: 102.01394011121278 ms

#### Problem 2

* Accuracy for KNN (threashold 0.9) on new faces: 0.9971

* Accuracy for Sequencial model on new faces: 0.9764

#### Problem 3

I tackled the Problem 3 using a clustering approach. 

The clustering algorithms are divided in two groups depending if the number of output clusters is known.
In my opinion in this case the user doesn't know the number of clusters, however this was not specified in the problem, therefore I implemented two clustering algorithms:
1. DBSCAN which automatically detects the number of clusters. 
2. Hierarchical clustering, which takes as input the number of clusters.

For the DBSCAN I need to provide the  maximum distance between two samples for them to be considered as in the same neighborhood.
For the best results this parameter is dependent of the data set. In order to provide it, I took advantage of the fact 
that I had some labeled images and I used the test set that I saved for the unittests and I computed the distance that is the most representative for the data set.
In case a labeled data set is not provided, this parameter should be detected by clustering using different values, and manually analyzing the resulting clusters.
  
When the final number of clusters is known, the algorithm used will be the hierarchical clustering.

* DBSCAN

Number of clusters: 163

Homogeneity: 0.968

Completeness: 0.980

V-measure: 0.974

Adjusted Rand Index: 0.967

Adjusted Mutual Information: 0.955

* Hierarchical

Number of clusters: 150

Homogeneity: 0.990

Completeness: 0.984

V-measure: 0.987

Adjusted Rand Index: 0.885

Adjusted Mutual Information: 0.978

#### The responses of the questions
*How does the number of people present effect the groupings?)

The perfect solution should create a cluster for each person present. 

*How many embeddings per person are needed for an effective group?*
The number of embedding per person affect only the DBSCAN algorithm. 
In this algorithm we have another parameter `min_samples' which sets how many embedding should have a person in order to create a cluster by itself.
I set it with 2.  

*How long does it take to run the grouping? Can it be done in real time?*

It took in on my computer 30987.141 ms. 
Real-time responses are often understood to be in the order of milliseconds, therefore, in my opinion, the clustering algorithm cannot be done in real time. 






## Scripts

* create_dataset.py - takes as input the set of images, and splits them in two subsets.
*-t* parameter states which percentage of images from each class will be used for testing
*-u*   parameter states which percentage of classes will be used only in the testing.

* train_sequencial_model.py - trains the sequential model. The output is written in the Config.SAVED_MODELS directory, versioned by day.

* train_knn_model.py - trains the knn model using the provided K value or by tuning first the model.

* problem1.py - solves the Problem 1

* problem2.py - solves the Problem 2

* problem3.py - solves the Problem 3

*Download the input dataset*

aws s3 cp s3://ii-ml/human-resources/tech-test-data-face.tar.gz resources/

tar -C resources/ -xzf resources/tech-test-data-face.tar.gz

*Split data for developing and unittesting* 

python src/create_dataset.py  -i resources/faces_org -o resources/data_tests/ -t 0.05 -u 0.05 -c 

mkdir  tests/fixtures/faces

mkdir tests/fixtures/faces/same_persons

mkdir tests/fixtures/faces/new_persons

mv resources/data_tests/test/* tests/fixtures/faces/same_persons

mv resources/data_tests/newdata/* tests/fixtures/faces/new_persons

rm -r resources/data_tests/test/

mkdir   resources/faces

mv resources/data_tests/train/* resources/faces

rm -r resources/data_tests/train/

rm -r resources/data_tests/

*Train the sequential model* 
python src/python src/train_sequencial_model.py -i resources/data_problem -b upsample

*Train the knn model*
python src/train_knn_model.py -i resources/data_problem -k 3

*Solve Problem 1 with the knn model*
python src/problem1.py -i resources/data_problem -m output/models/2018-10-22 -a knn

*Solve Problem 1 with the sequential model*

python src/problem1.py -i resources/data_problem -m output/models/2018-10-22 -a seq

*Solve Problem 2 with the knn model*

python src/problem2.py -i resources/data_problem -m output/models/2018-10-22 -a knn

*Solve Problem 2 with the sequential model*

python src/problem2.py -i resources/data_problem -m output/models/2018-10-22 -a seq

*Solve Problem 3 with the DBSCAN approach*

python src/problem3.py -i resources/faces_org -o output/clusters

*Solve Problem 3 with the Hierarchical approach*

python src/problem3.py -i resources/faces_org -o output/clusters_150 -n 150
