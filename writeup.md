## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_imgs/hog_feat.jpg
[image2]: ./writeup_imgs/hog_feat_og.jpg
[image3]: ./writeup_imgs/windows.jpg
[image4]: ./writeup_imgs/3_sample_test.jpg
[image5]: ./writeup_imgs/2_sample_test.jpg
[image6]: ./writeup_imgs/decision_function_hist.jpg
[image7]: ./writeup_imgs/1_heat_map.jpg
[image8]: ./writeup_imgs/2_heat_map.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First, to identify the best set of features to train the Support Vector Machine model, I visualize each the color histograms and the color spaces on a 3d plot for the RGB and HSV color space. These visualizations can be found on the first nine cells of the jupyter notebook.
After visualizing the HSV color space I discover that Saturation and Value are the features that can be used to better represent an image given its distribution values.

For the exploration stage I used a single image from the test images.

Next, I decided to visualized its histogram of gradients using the value channel. The functions that compute the histogram of gradients are located on the block number 9 from the jupyter notebook.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

From the beginning I did not focused on testing out the HOG parameters and I decided to keep the default values from function developed during the Udacity lesson. 

I started to modify the parameters when I feed the features into the model. In this case I trained a SVM on the small car/non-car dataset for faster iterations. 

During this processed I noticed that the Value channel from the HSV color space performed a bit better than the Saturation channel and I decided to keep it. Since I did not need a lot of definition from the gradients I decided to keep the 9 orientations, 8 pixels per cel and 2 cell_per_block.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear support vector machine using the scikit learn LinearSVC() class. The features get extracted on the cell #15 and the training stage exists on cell #16. The feature extraction is done by calling the `extract_features` function on cell #14 which reads the whole set of car and non_car images and calls `single_img_features` to obtain the images features one by one. 
For the feature extraction step I decided to use the hog features, spatial features and histogram of the HSV color space.

The whole dataset was first split on a training set containing 80% and a testing set with 20% of the images. 
In order to normalize the image feature vector, I use the scikit learn `StandardScaler()` class.

After the training stage is done, I decided to test the model with the testing set obtaining an accuracy of %95.69.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the window searching I decided to add several window rows at different scales. I started by creating a row of windows of 20px by 20 px and place them in the center of the horizon where cars on a small scale are visible. After that I manually increase the scale and window placement of a second row of windows, this time aiming for cars at a bigger size. I repeated this process with the 2, 4, 5, 6, 7, 8, 10, 12 scale factors. At each of the scales the start and end place on the y axis was also modified to make the window appear on the center of the horizon even though its size increased. This modification was implemented using the `y_shift` variable on the python notebook. I also overlapped the windows at a 40%.
The code for sliding windows is located on the 17th cell on the notebook
Finally, I tested the windows using the images on the `test_images` folder.

Here is an example of how the windows look:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As mentioned before for the classifier training I decided to use the hog features, spatial features and histogram of the HSV color space. These features gave me good performance when searching on the sliding windows showed above. To improve the performance of the classifier on false positives I used the `decision_function` method to add an additional threshold, more details are explained on the next section. 
Most of the pipeline code can be found on the `FindCar` class and can be divided on the following stages:
- Getting the hot windows by calling the `search_windows` function, which runs the classifier on every window. 
- Adding the hot window results to the rolling window array.
- Applying heat threshold to the overall all of the windows on rolling window array. (the hot windows retention is defined by the `hot_win_retention` on the class)
- Finally the valid car windows are then drawn into the current frame.

Here are some example images:

![alt text][image4]
![alt text][image5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
I uploaded my video to youtube and can be found here: https://youtu.be/ft_8dBt5vrk


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

First, I noticed that in some occasions I was getting rectangular boxes that looked like columns, after applying the heatmap threshold to the matched boxes. I decided to remove them by checking its dimensions and making sure that the width of the box was bigger than its height. This is done on the `is_valid_car_box` function.

Second, I was getting a lot of false positives at random scales and places on my first run. I decided to look at the documentation to find a classifier output that can give me the level of confidence on the classification step rather than a binary value. I found that the `SVC` `decision_function` method returns the distance from the sample points to the multi dimensional hyperplane. I was able leverage the output of the `decision_function` to built an additional threshold that can serve as a stronger filter for false positives. In order, to better understand the output of this new functions I decided to create a histogram from the results of applying the classifier `decision_function` on the test images sliding windows. 
Here is the histogram plotted:
 ![alt text][image6]
 
After looking at the histogram I decided to set the threshold to 450 since it give me a good balance between removing the false positives while at the same time captures most of the highly confident prediction. This threshold performed the best when I tested the pipeline on the video stream. 

Finally, in order to only keep the most highly confident classifications I decided to keep the positive classified windows on a rolling window array. This rolling window keep track of the positive classifications over a certain amount of frames. I apply the heatmap and threshold on the whole rolling window array and increase the threshold overlap to 15. 

Here is are some images of the heatmaps on test images:
 ![alt text][image7]
 ![alt text][image8]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of the issues that I faced on this project were related to filtering false positives as described on the previous section. However, I would like to add that throughout the project my other biggest issue was to stabilization and location of the car boxes. I believe that this was caused by some false positives that made it through the pipeline and the car position differences on each frame.
I think that false positive getting solved with a more accurate classifier using advanced techniques such as deep learning, however this might cause the pipeline to be slower. In addition, finding the correct sliding windows on an image was a hard problem for me. There were times were the classifier had a hard time identifying an image due incorrect window position. This problem can also be resolved with some other advanced deep learning techniques, where the windows selection and classification can happen on the same step.
Also, cars tend to move on a particular direction on the video stream. Adding additional information to predict where the car might be moving given the previous frames can result on a huge perfornace improvement. 

Finally, I found that my pipeline completely failed when the road changed colors. This might be because of the way I trained the classifier and the data used to train it. I believe that a more robust training dataset and selecting more general features can help on making the classifer more robust.