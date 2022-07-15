#June 30 2022
"""
import  https://www.kaggle.com/datasets/jeffheaton/iris-computer-vision dataset to your notebook"""

import cv2 as cv

%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

iris_img = cv.imread('../input/iris-computer-vision/iris-virginica/iris-08c7a0e53ee1de193062d7bf86a21adec67b9115fd6099f1b2d2ee728a936002.jpg')

#cv.imshow(iris_img)

img_plot = plt.imshow(iris_img)
plt.show()
"""
first ir reads the iris image and then img_plot stores the iris image and plt prints out the irist image"""

# Accessing Individual Pixels
(b, g, r) = iris_img[0, 0]
print(b,g,r)
""" gives the ratio of blue and green and red in a pixel"""

# Array/Region of Interest (ROI) cropping

pink = iris_img[80:200, 80:200]
plt.imshow(pink)
"""
this stores a region of 80 to 200 on x and 80 to 200 on y in a variable called pink and prints it out"""

# Resizing images

print(iris_img.shape)
down_width = 200
down_height = 200
down_points = (down_width, down_height)
resized_down = cv.resize(iris_img, down_points)
plt.imshow(resized_down)
"""
this is to resize an image to 200 on both x and y axis"""

# Rotating an image

# https://www.kaggle.com/datasets/aditipanda1/exprotation
# https://learnopencv.com/image-rotation-and-translation-using-opencv/

# grab the dimensions of the image and calculate the center of the image
(h, w) = iris_img.shape[:2]
center = (w // 2, h // 2)

# rotate our image by 45 degrees around the center of the image
rotate_matrix = cv.getRotationMatrix2D(center=center, angle=45, scale=1)

rotated = cv.warpAffine(src=iris_img, M=rotate_matrix, dsize=(w, h))

plt.imshow(rotated)
"""
this roates the image by 45 degrees. first you find the center of the image and then you do getroration to get a rotational matrix and then
you rorate the image with warpaffine with the source and then the rotational matrix and then center"""




# now for image analysis
"""
What is edge detection? Edge detection is an image processing technique for finding the boundaries of objects within images. It works by detecting 
discontinuities in the image brightness"""

# Convert to graycsale
img_gray = cv.cvtColor(iris_img, cv.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray')
"""
this converts iris_img to colorBGR2GRAY and then plt prints out img_gray with a cmap also known as color map as gray"""

# Blur the image for better edge detection

img_blur = cv.GaussianBlur(img_gray, (3,3), 0)
plt.imshow(img_blur, cmap='gray')
"""
to blur the params for gaussianblur is the src file and then the ksize or also known as the kernel size of height and width for the grain of blur. higher
the ksize the larger and more blurred the image then sigma x which give std along x axis and then next is sigma y"""

# Sobel Edge Detection
# https://learnopencv.com/edge-detection-using-opencv/

sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
plt.imshow(sobelx, cmap='gray')

"""
what is sobel edge detection? It is one of the widely used algorithms for edge detection. The sobel operator detects edges that are marked by sudden change 
in pixel intensity and then we plot the first derivative where you can see the change in intensity and where it is 0.

There are 2 kernels for sobel edge detection(2 matrix) which are x direction kernel and y direction kernel. These kernels when used on the original image it
transforms the image to get a sobel edge image.
If we use only the vertical kernel the edges are enhanced in the x direction
If we use only the horizontal kernel the edges are enhanced in the y direction

Syntax:
sobel(src, ddepth, dx, dy)
src is the source of the image
ddepth specifies the precision of the output image
dx and dy specify the order of the derivative in each direction
dy = 1 and dx = 0 means compute image in 1st derivative of sobel image in y direction
if both are 1 then it computes the 1st derivative in both x and y direction
after dy is ksize which represents the size of the kernel"""

sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
plt.imshow(sobely, cmap='gray')
"""this is for dy=1 """

sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) 
plt.imshow(sobelxy, cmap='gray')
""" this is in x and y direction """

edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200)
plt.imshow(edges, cmap='gray')
""" this is to give the edges in the picture
Canny is another popular edge detection algorithm it can extract edges from a blurred image
first it reduces the noise
then calculates the intensity
supresses the false edges
then hysteresis thresholding

image is the blurred image
raising the threshold gives lesser edges
raising threshold2 decreases edges shown faster than raising threshold1
lower threshold2 is the more edges shown

Hysteresis Thresholding
In this final step of Canny Edge Detection, the gradient magnitudes are compared with two threshold values, one smaller than the other. 

If the gradient magnitude value is higher than the larger threshold value, those pixels are associated with strong edges, and are included in the final edge map.
If the gradient magnitude values are lower than the smaller threshold value, the pixels are suppressed, and excluded from the final edge map.
All the other pixels, whose gradient magnitudes fall in between these two thresholds, are marked as ‘weak’ edges (i.e. they become candidates for being included in the final edge map). 
If the ‘weak’ pixels are connected  to those associated with strong edges, then they too are included in the final edge map. 

"""

img = iris_img
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
"""
this gives various images with changes in thresholds. it is a type of image segmentation where we change the pixels of an image to make the image 
easier to analyse. we can manipulate certain parts of the image while leaving the rest of it out.
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html#:~:text=value%20is%20calculated%3A-,cv.,values%20minus%20the%20constant%20C.

first argument is the source image
second argument is the threshold value which is used to classify the pixel value
third argument is the maximum value assigned to the pixel which exceeds the threshold
fourth argument is to give the type of thresholding to be done

this then returns the threshold which was used and then the thresholded image

to plot multiple images then use plt.subplot"""

new_iris_img = cv.imread('../input/iris-computer-vision/iris-versicolour/iris-009322547cb81cbb4dbc63c298304a2df8ea617191651d4cbd50b7a5bbba8a6a.jpg', 0)

plt.imshow(new_iris_img, cmap='gray')
""" 
new iris image"""

img = cv.medianBlur(new_iris_img,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
"""
simple thresholding uses one value throughout the image while adaptive thresholding determines the threshold for a pixel based on a small region around it
which is useful for images withdifferent lighting conditions in different areas

cv.ADAPTIVE_THRESH_MEAN_C: The threshold value is the mean of the neighbourhood area minus the constant C.
cv.ADAPTIVE_THRESH_GAUSSIAN_C: The threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C.

blockSize determines the size of the neighbourhood area and C is a constant that is subtracted from the mean or weighted sum of the neighbourhood pixels.
"""

"""
Masking and Bitwise Ops
In computer science terms, everything we represent is converted into binary language and also stored in binary format.

When it comes to displaying an image if there is no color in the pixel, the value is assigned ‘0’, and when there is some color stored in the pixel then
the value is assigned ‘1’.
When we perform bitwise operations on images in OpenCV, they are actually performed on 0’s and 1’s of the image."""

import numpy as np
black = np.zeros((480, 480), dtype = "uint8")
plt.imshow(black, cmap='gray')
"""
a pure black image is shown"""

black = np.zeros((480, 480), dtype = "uint8")
rect = cv.rectangle(black.copy(), (50, 60), (400, 200), 255, -1)
circle = cv.circle(black.copy(), (240, 240), 150, 255, -1)
bit_and = cv.bitwise_and(rect, circle)
plt.imshow(rect, cmap='gray')
plt.imshow(circle)
plt.imshow( bit_and, cmap='gray')
"""
rect gives the rectange with both of the opposite points of the rectange and circle has the center point of the circle and the next number is the radius. 
bitwise and is done and part of the circle is shown in white which is the part where the rect and circ overlap each other"""

bit_or = cv.bitwise_or(rect, circle)
plt.imshow( bit_or, cmap='gray')
"""
both circle and rect are shown in white overlapping each other and their respective areas are white"""

bit_Xor = cv.bitwise_xor(rect, circle)
plt.imshow( bit_Xor, cmap='gray')
"""
their intersection is black while their respective areas are white"""

bit_not1 = cv.bitwise_not(rect)
plt.imshow( bit_not1, cmap='gray')
"""
backround is white while rect is black"""

bit_not2 = cv.bitwise_not(circle)
plt.imshow( bit_not2, cmap='gray')
"""
backround is white while full circle is black"""

## Masking

#create a blank image using np.zeros()
blank = np.zeros(iris_img.shape[:2], dtype = "uint8")
print(blank.shape)

#draw rectangle on the blank image
mask = cv.rectangle(blank, (100, 150), (120, 200), (255, 0, 255), -1)
plt.imshow( mask, cmap='gray')

maskedimage= cv.bitwise_and(iris_img, iris_img, mask=mask)
plt.imshow( maskedimage)
"""
this shows a part of the iris image inside the rectangle which is drawn on the blank image"""

B, G, R = cv.split(iris_img)
plt.imshow(B)
"""
this seperates the image into three color channels"""

im1 = cv.imread('../input/iris-computer-vision/iris-virginica/iris-3b19970f2f6807b8a41c3c426d4b0032fba84d025929eb6e6ff01501995fc1c2.jpg')
plt.imshow(im1)
im1 = cv.resize(im1, (180,180))
plt.imshow(im1)
im2 = cv.imread('../input/dummy-astronomy-data/Cutout Files/star/IC745_H01_1459_1334_6.jpg')
print(im2.shape)
im2 = cv.resize(im2, (180,180))
plt.imshow(im2)
weightedSum = cv.addWeighted(im1, 0.5, im2, 0.4, 0)

plt.imshow(weightedSum)
sub = cv.subtract(im1, im2)
plt.imshow(sub)
"""
here we take 2 images, one is a flower image while the other is an astronmical image. We resize both to 100 pixel square then we show the image when we add
both of them together and both when we subtract them"""


"""
Histogram Computation"""

# grayscale histogram
hist = cv.calcHist([img_gray], [0], None, [256], [0, 256])
plt.plot(hist)

"""
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

Parameters:

images: list of images as numpy arrays. All images must be of the same dtype and same size.
channels: list of the channels used to calculate the histograms.
mask: optional mask (8 bit array) of the same size as the input image.
histSize: histogram sizes in each dimension
ranges: Array of the dims arrays of the histogram bin boundaries in each dimension
hist: Output histogram
accumulate: accumulation flag, enables to compute a single histogram from several sets of arrays.

here 0 is blue channel 0r grayscale
1 is green
2 is red channel

x axis is the pixel value of that color from 0 to 255
y axis is the number of pixels 
0 is dark 255 is white"""

# color channel wise histograms

chans = cv.split(iris_img)
colors = ("b", "g", "r")

for (chan, color) in zip(chans, colors):
    
	# create a histogram for the current channel and plot it
	hist = cv.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color=color)
	plt.xlim([0, 256])
  
"""
iris_img split into three channels and the histogram for them is shown"""

fig = plt.figure()
ax = fig.add_subplot(131)
hist = cv.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
#ax.set_title("2D Color Histogram for G and B")
#plt.colorbar(p)

ax = fig.add_subplot(132)
hist = cv.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
#ax.set_title("2D Color Histogram for G and R")
#plt.colorbar(p)

ax = fig.add_subplot(133)
hist = cv.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
#ax.set_title("2D Color Histogram for B and R")
#plt.colorbar(p)

plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)

#print("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))

"""
Resources:

https://learnopencv.com/image-rotation-and-translation-using-opencv/

https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/"""





"""
Time for practice question!
Perform the following steps:

Go to "Add data" button on top right corner of your notebook and search for this url "https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs"
Click add to import the "cityscapes-image-pairs"
Run below code cell"""

import os

root_dir = "../input/cityscapes-image-pairs/cityscapes_data/train"

sample = os.path.join(root_dir, "101.jpg")

"""
Q-1. Let's detect the edge for the above sample image i.e. "101.jpg" from the cityscapes-image-pairs datasets."""

# perform all necesaary steps and detect edge for an 101.jpg image.
import cv2 as cv

%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

city_img = cv.imread('../input/cityscapes-image-pairs/cityscapes_data/train/101.jpg')
plt.imshow(city_img)

#1ets cr0p the ph0t0
(h, w) = city_img.shape[:2]
print(w)
w = int(w/2)
city2 = city_img[0:256, 0:w]
plt.imshow(city2)

#n0w grayscale and blur the image
city_gr = cv.cvtColor(city2, cv.COLOR_BGR2GRAY)
plt.imshow(city_gr, cmap='gray')

city_br = cv.GaussianBlur(city_gr, (3,3), 0)
plt.imshow(city_br, cmap='gray')

#sobel edge detection in both x and then y direction
sobelx = cv.Sobel(src=city_br, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv.Sobel(src=city_br, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
images = [sobelx,sobely]
titles = ['sobelx', 'sobelx']
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i], 'gray')
    plt.title(titles[i])
plt.show()

#now for canny edge detection
edges = cv.Canny(image=city_br, threshold1=50, threshold2=51)
plt.imshow(edges, cmap='gray')


"""
Q-2. Rotate the above image taken in "sample" at the angle of 40."""

# Rotate an image at an angle of 40.
(h, w) = city_img.shape[:2]
center = (w//2, h//2)
rt_m = cv.getRotationMatrix2D(center=center, angle=40, scale=1)
rt = cv.warpAffine(src = city_img, M = rt_m, dsize=(w,h))
plt.imshow(rt)



"""
Q-3. A company named MNP have 2 products facewash and shampoo

Months = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November','December']

Facewash = [1500 , 1200, 1340, 1130, 1740, 1555, 1120, 1400, 1780, 1890, 2100, 1760]

Shampoo = [1200 , 2100, 3550, 1870, 1560, 1890, 1780, 2860, 2100, 2300, 2400, 1800]

Moisturizer = [1500, 1200, 1340, 1130, 1740, 1555, 1120, 1400, 1780, 1890, 2100, 1760]

Plot the subplot for each product mentioned above. Take sales for the product on the y-axis and month of the sale on x-axis. 
So in total there are three subplots. Also apply grid in the graph, write the title of the graph, assign names to x-axis and y-axis, 
add legends and xticks to the x-axis."""

# Plot subplots here in single cell only.
Months = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November','December']
Facewash = [1500 , 1200, 1340, 1130, 1740, 1555, 1120, 1400, 1780, 1890, 2100, 1760]
Shampoo = [1200 , 2100, 3550, 1870, 1560, 1890, 1780, 2860, 2100, 2300, 2400, 1800]
Moisturizer = [1500, 1200, 1340, 1130, 1740, 1555, 1120, 1400, 1780, 1890, 2100, 1760]

plt.figure(figsize=(40, 5))
plt.subplot(131)
plt.xlabel('Months')
plt.ylabel('Facewash')
plt.plot(Months, Facewash)
plt.subplot(132)
plt.xlabel('Months')
plt.ylabel('Shampoo')
plt.plot(Months, Shampoo)
plt.subplot(133)
plt.xlabel('Months')
plt.ylabel('Moisturizer')
plt.plot(Months, Moisturizer)
plt.show()




