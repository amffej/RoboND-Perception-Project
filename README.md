[//]: # (Image References)
[image1]: ./images/intro.png
[image2]: ./images/intro2.png
[image3]: ./images/amazon.jpg
[image4]: ./images/conf_matrix_n100.png
[image5]: ./images/conf_matrix_normalized_n100.png
[image6]: ./images/world_1.png
[image7]: ./images/world_2.png
[image8]: ./images/world_3.png
## Project: 3D Perception Pick & Place
![alt text][image1]

This project is based on the Amazon Robotics’ Challenge. The main task is to give a robot the ability to locate an object in a cluttered environment, pick it up and then move it to some other location. This is not just an interesting problem to try to solve, it's a challenge at the forefront of the robotics industry today. To solve this problem, we use point cloud analysis, filtering, clustering segmentation, image histogram analysis and SVM (Support vector machine) a machine learning technique that uses supervised learning models with associated algorithms to analyze data for classification.
This project is set up and completed in a simulated environment. We use ROS, Gazebo, and Rviz MoveIt to complete all the required functions. This project uses a PR2 robot that has been outfitted with an RGB-D. To make things much like the real world, this sensor comes default with a bit of noise on its output stream.
We are given a cluttered tabletop scenario; the goal is to implement a perception pipeline that can identify target objects from a "Pick-list" in a specific order and pick and place those objects in their corresponding drop-boxes.
![alt text][image3]
---
### Exercise 1. Filtering pipeline and RANSAC plane filtering
For this exercise the following filters were applied to the source point cloud
-   VoxelFrid Down sampling
-   PassThrough
-   Extract Indices
-   RANSAC Plane Fitting
-   Statistical Outlier Removal
### Exercise 2. Clustering and segmentation
For this exercise the following items were accomplished
-   Applied Euclidian clustering on filtered point cloud
-   Published resulting cloud to separate topics
### Exercise 3. Feature Extraction and SVM classifier training for object recognition
For this exercise we use their color and shape to classify and recognize them. We then apply labels for visualization purposed.
-   Feature extractions performed using `capture_features.py`
    - We convert RGB to HSV then numpy to create color histograms
- Applied SVM (Support Vector Machines) classification using `train_svm.py`     

### Discussion
The result of the `train_svm.py` can be overserved on the confusion matrix below. In order to get to this acurazy the iteration cycle was modified to 100 iteration per object. This process took aproximately 1 hours.
![alt text][image4]
![alt text][image5]
Success rate for world 1 is 100% all objects were successfully classified
![alt text][image6]
Success rate for world 2 is 100% all objects were successfully classified
![alt text][image7]
Success rate for world 3 is 87% 7/8 objects were successfully classified. The glue bottle was not successfully classified, this can be a result of the book partially blocking this object. 
![alt text][image8]
