#July 1 2022
# image

from PIL import Image

import warnings
warnings.filterwarnings('ignore')
im = Image.open('../input/film-image/elephant.jpg')
print(im.getbands())
print(im.mode)
print(im.size)
im

box = (200, 0, 300, 300)
cropped_image = im.crop(box)
cropped_image

rotated = im.rotate(180)
rotated

im_1 = Image.open('../input/film-image/film.png')
position = (200, 400)
im.paste(im_1, position)
im.save('merged.jpg')
im

im = Image.open('../input/film-image/elephant.jpg')
image_copy = im.copy()
position = (150,400)
image_copy.paste(im_1, position)
image_copy

im.transpose(Image.FLIP_TOP_BOTTOM)

import numpy as np
im_array = np.array(im)

im.convert('1')
im

im.convert('RGBA')
im

from PIL import ImageDraw
image = Image.new('RGB', (400, 400))
img_draw = ImageDraw.Draw(image)
img_draw
img_draw.rectangle((100, 30, 300, 200), outline='red', fill='white')
img_draw.text((150, 100), 'red-rectangle', fill='red')
image.save('drawing.jpg')
drawing = Image.open('./drawing.jpg')
drawing

from PIL import ImageEnhance
fox = Image.open('../input/film-image/fox.jpg')
enhancer = ImageEnhance.Sharpness(fox)
enhancer.enhance(10.0)
enhancer = ImageEnhance.Contrast(fox)
enhancer.enhance(2)

from PIL import ImageFilter
fox.filter(ImageFilter.BLUR)
fox.thumbnail((100,100))
fox



"""
Scikit-learn
Basics of ML - Supervised, Unsupervised, Reinforcement learning

Supervised learning:
Uses data which is labelled both input and corresponding output to understand their interaction and allow it to predict future behavior depending 
on the past behavior taken from the data

there are 2 ways in which the output is given:
- continuous which gives values like example stock market predicitions also this is known as regression
- or gives discrete values such as true or false or a binary digit or a value from a discrete system

Unsupervised learning:
Data is given which is unlabelled and the computer groups similar data together and finds the relatioship between them and establishes it

Reinforcement learning:
Data comes fro as we go about the process and it learns what to do and what not to do from the feedback given as it completes the task
examples are chess games or learning to play a new video game"""

"""
What is logistic regression?
- it is a way of binary classification
-ski kit has all these algorithms"""

import pandas as pd
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1599 entries, 0 to 1598
Data columns (total 12 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         1599 non-null   float64
 1   volatile acidity      1599 non-null   float64
 2   citric acid           1599 non-null   float64
 3   residual sugar        1599 non-null   float64
 4   chlorides             1599 non-null   float64
 5   free sulfur dioxide   1599 non-null   float64
 6   total sulfur dioxide  1599 non-null   float64
 7   density               1599 non-null   float64
 8   pH                    1599 non-null   float64
 9   sulphates             1599 non-null   float64
 10  alcohol               1599 non-null   float64
 11  quality               1599 non-null   int64  
dtypes: float64(11), int64(1)
memory usage: 150.0 KB

quality is the dependent variable which we will predict because the quality is dependent on all other factors"""

y = df.quality
x = df.drop('quality', axis=1)
"""
separating independent and dependent"""

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
"""
now you have to split the data into training set data and then testing set data, test_size tells how much of the set is going to train the set
then compares the trained and then the test actual value
"""
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train, y_train)
y_pred = reg.predict(x_test)
"""
now import the linear regression algorithm and we run it on the trained data and the output it gives to give a prediction model
then predict y from the x_test data against the made prediction"""

from sklearn.metrics import mean_squared_error,r2_score
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
"""
now compare the predicted y and the actual test y and find the mean squared error"""

from joblib import dump, load
dump(reg, 'WineLinReg.joblib') 
"""
save the made prediction model which you can use late on"""
