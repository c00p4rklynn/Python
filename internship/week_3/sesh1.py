"""
Linear regression:
Supervised Learning:

Input data --> Corresponding lables as well

y = m*x + c

m = slope and c = intercept

output = weight * input + bias

for multiple variables:
output = w1x1 + w2x2 + w3*x3 + b

x parameters are input parameters and that we already have with us y is dependent on x and w and b

w and b are : model parameters

now we are creating a star classification of size from brightness of a star:"""

import numpy as np
import matplotlib.pyplot as plt

#now create a set of random data of brightness

X = 3*np.random.rand(100,1)
y = 9 + 2*X + np.random.rand(100,1)

# now a scatter plot of x and y linear wise
plt.scatter(X,y)
plt.xlabel('Brightness')
plt.ylabel('Star size')
plt.title('Brightness of Star vs Size of Star')
plt.figure(figsize=(15,25));

# getting the threshold

thresh = np.median(y)
thresh

# now we are concluding that if size of star is bigger than thresh, then it is a BIG star
# otherwise it's a SMALL star
#12.785090363059805

m, n = X.shape
m, n
#(100, 1)

"""
(y) = output = W*x +b
w is weight
b is bias

w,b = 0

y_cap = w*X1 + b 
y = 10.07

y_cap = 12

find the error between predicted and actual output:

Error = (y - y_cap)

cost_function = Mean Square Error ( Error):

cost_func = C_f = 1/N * (summation (y-y_cap)**2)
y_cap = W*x + b

w, b in order to change your output.

minimize the error and update the values of weights and bias
after getting the derivative of weight and bias

we will update the weights and bias ⁉

w_updated = weight_old - learning_rate(derivative of weight) 
b_updated = bias_old - learning_rate (derivative of bias)

by updating weight and bias we get a more accurate model with more accurate weight and bias
"""

W = np.zeros(n)
b = 0
# first making weight and bias 0

iterations = 1000
learning_rate = 0.01
# how many iterations we are going to through to train the model

def predict(X, W, b):
    return W*X + b
#y = mx+c


def cost_function(X, y, W, b):
    n = len(X)
    total_error = 0.0
    for i in range(n):
        total_error += (y[i] - (W*X[i] + b))**2
    return total_error / n
# now we make a cost function to help minimize error
# we calculate using mean square error which is change in y then square it and then sum are iteration of this and then divide by number of iterations

""" 
https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html#gradient-descent
"""

"""
Next we find gradient descent which is finding the gradient of our cost function to help minimize the error
Gradient descent consists of looking at the error that our weight currently gives us, using the derivative of the cost function to find the gradient and 
thus changing the weight such that it opposes the current gradient of the function thus aims for an equillibrium where the gradient is 0

Thus we know cost_func = C_f = 1/N * (summation ((y - (W*X + b))**2)
let:
A(B(w,b)) = (y - (W*X + b))**2
A(x)=x**2
df/dx=A′(x)=2x

B(w,b) = y - (W*X + b)
dx/dw = B'(w) = -x
dx/db = -1

thus chain rule to find the derivative of cost function with respect to weight:
df/dw = df/dx * dx/dm
= 2(y−(wx+b))*−x

thus chain rule to find the derivative of cost function with respect to bias:
df/db = df/dx * dx/db
= 2(y−(wx+b))*−1

Thus gradient is:

f'(w, b) = [1/N * (summation -2x(y−(wx+b))) , 1/N * (summation -2(y−(wx+b)))]

Now to solve for the gradient:
-we iterate through the number of data points in X which is star brightness
-using the weight and bias we currently have we input it into it
- we take the average partial derivatives at each data point telling us the sharpest slope of ascent and multiply it by learning rate
  learning rate: it is the size of updating during a optimization loop like update weights which is also known as gradient descent
    With a high learning rate we can cover more ground each step, but we risk overshooting the lowest point since the slope of the hill 
    is constantly changing. With a very low learning rate, we can confidently move in the direction of the negative gradient since we are 
    recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is time-consuming, so it will take us a very 
    long time to get to the bottom.
- then subtract it from the current weight we have and we keep on repeating as it optimizes the weight and bias with each loop of each data point"""

def update_weights(X, y, W, b, learning_rate):
    dW = 0
    db = 0
    n = len(X)

    for i in range(n):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        dW += -2*X[i] * (y[i] - (W*X[i] + b))

        # -2(y - (mx + b))
        db += -2*(y[i] - (W*X[i] + b))

    # We subtract because the derivatives point in direction of steepest ascent
    W -= (dW / n) * learning_rate
    b -= (db / n) * learning_rate

    return W, b
  
"""
next is training the model:
-this is improving your prediction model by iterating through the data set multiple times and each time updating the weight and bias in the direction opposite
of the slope of the cost function.
-Training is complete when we reach an acceptable error threshold, or when subsequent training iterations fail to reduce our cost."""

def train(X, y, W, b, learning_rate, iterations):
    costs = []

    for i in range(iterations):
        W, b = update_weights(X, y, W, b, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(X, y, W, b)
        costs.append(cost)

        # Log Progress
        if i % 10 == 0:
            print("iter={}, weight={}, bias={}, cost={}".format(i, W, b, cost))

    return W, b, costs
  
costs = np.zeros(iterations)

# run train
W, b, costs = train(X, y, W, b, learning_rate, iterations)
"""
iter=0, weight=[0.37485299], bias=[0.24548095], cost=[134.20680936]
iter=10, weight=[2.90564008], bias=[2.00295449], cost=[38.66752482]
iter=20, weight=[4.06677504], bias=[2.97148936], cost=[15.66375795]
iter=30, weight=[4.56639583], bias=[3.55292646], cost=[9.75468587]
iter=40, weight=[4.74775513], bias=[3.94228512], cost=[7.90307289]
iter=50, weight=[4.77718796], bias=[4.23429623], cost=[7.0409276]
iter=60, weight=[4.73525134], bias=[4.47505723], cost=[6.44423567]
iter=70, weight=[4.66091485], bias=[4.68707817], cost=[5.94025745]
iter=80, weight=[4.57298711], bias=[4.8814261], cost=[5.48605107]
iter=90, weight=[4.48050476], bias=[5.06361373], cost=[5.06936921]
iter=100, weight=[4.38776952], bias=[5.23645486], cost=[4.68534625]
iter=110, weight=[4.29679046], bias=[5.40144919], cost=[4.33100479]
iter=120, weight=[4.20846802], bias=[5.5594539], cost=[4.00395238]
iter=130, weight=[4.12316796], bias=[5.71100957], cost=[3.70206427]
iter=140, weight=[4.04099962], bias=[5.85649833], cost=[3.4233988]
iter=150, weight=[3.96195063], bias=[5.99622081], cost=[3.16616831]
iter=160, weight=[3.8859522], bias=[6.13043358], cost=[2.92872371]
iter=170, weight=[3.81291065], bias=[6.2593675], cost=[2.70954302]
iter=180, weight=[3.74272256], bias=[6.38323678], cost=[2.5072214]
iter=190, weight=[3.67528211], bias=[6.50224358], cost=[2.32046206]
iter=200, weight=[3.61048448], bias=[6.61658031], cost=[2.14806797]
iter=210, weight=[3.54822742], bias=[6.726431], cost=[1.98893419]
iter=220, weight=[3.48841198], bias=[6.83197201], cost=[1.84204074]
iter=230, weight=[3.43094269], bias=[6.9333726], cost=[1.70644611]
iter=240, weight=[3.3757277], bias=[7.0307953], cost=[1.58128123]
iter=250, weight=[3.32267865], bias=[7.12439618], cost=[1.46574385]
iter=260, weight=[3.27171061], bias=[7.2143252], cost=[1.35909343]
iter=270, weight=[3.22274196], bias=[7.30072642], cost=[1.2606464]
iter=280, weight=[3.17569428], bias=[7.38373821], cost=[1.16977178]
iter=290, weight=[3.13049221], bias=[7.46349357], cost=[1.08588709]
iter=300, weight=[3.08706336], bias=[7.54012022], cost=[1.0084547]
iter=310, weight=[3.04533817], bias=[7.6137409], cost=[0.93697829]
iter=320, weight=[3.0052498], bias=[7.68447354], cost=[0.87099974]
iter=330, weight=[2.96673405], bias=[7.75243143], cost=[0.81009616]
iter=340, weight=[2.92972922], bias=[7.81772342], cost=[0.7538772]
iter=350, weight=[2.89417604], bias=[7.88045409], cost=[0.70198251]
iter=360, weight=[2.86001757], bias=[7.94072391], cost=[0.65407949]
iter=370, weight=[2.82719909], bias=[7.99862943], cost=[0.6098611]
iter=380, weight=[2.79566804], bias=[8.05426339], cost=[0.56904392]
iter=390, weight=[2.76537391], bias=[8.1077149], cost=[0.53136634]
iter=400, weight=[2.73626818], bias=[8.15906958], cost=[0.49658687]
iter=410, weight=[2.70830423], bias=[8.20840968], cost=[0.46448258]
iter=420, weight=[2.68143727], bias=[8.25581423], cost=[0.4348477]
iter=430, weight=[2.65562426], bias=[8.30135916], cost=[0.4074923]
iter=440, weight=[2.63082387], bias=[8.34511742], cost=[0.38224103]
iter=450, weight=[2.60699636], bias=[8.38715911], cost=[0.35893206]
iter=460, weight=[2.58410358], bias=[8.42755155], cost=[0.33741597]
iter=470, weight=[2.56210885], bias=[8.46635945], cost=[0.31755487]
iter=480, weight=[2.54097694], bias=[8.50364496], cost=[0.29922146]
iter=490, weight=[2.52067402], bias=[8.53946781], cost=[0.28229823]
iter=500, weight=[2.50116755], bias=[8.57388538], cost=[0.2666767]
iter=510, weight=[2.48242629], bias=[8.60695279], cost=[0.25225677]
iter=520, weight=[2.46442023], bias=[8.63872301], cost=[0.23894599]
iter=530, weight=[2.44712052], bias=[8.66924692], cost=[0.22665906]
iter=540, weight=[2.43049946], bias=[8.69857342], cost=[0.21531722]
iter=550, weight=[2.41453043], bias=[8.72674948], cost=[0.20484777]
iter=560, weight=[2.39918783], bias=[8.75382023], cost=[0.19518363]
iter=570, weight=[2.38444711], bias=[8.77982903], cost=[0.18626283]
iter=580, weight=[2.37028465], bias=[8.80481753], cost=[0.17802821]
iter=590, weight=[2.35667776], bias=[8.82882577], cost=[0.17042698]
iter=600, weight=[2.34360465], bias=[8.8518922], cost=[0.16341043]
iter=610, weight=[2.33104438], bias=[8.87405376], cost=[0.15693358]
iter=620, weight=[2.31897684], bias=[8.89534596], cost=[0.15095492]
iter=630, weight=[2.30738269], bias=[8.91580288], cost=[0.14543613]
iter=640, weight=[2.29624336], bias=[8.93545731], cost=[0.14034184]
iter=650, weight=[2.28554102], bias=[8.95434072], cost=[0.1356394]
iter=660, weight=[2.27525851], bias=[8.97248336], cost=[0.13129865]
iter=670, weight=[2.26537938], bias=[8.98991428], cost=[0.1272918]
iter=680, weight=[2.25588779], bias=[9.00666142], cost=[0.12359314]
iter=690, weight=[2.24676854], bias=[9.02275158], cost=[0.12017898]
iter=700, weight=[2.23800703], bias=[9.03821055], cost=[0.11702743]
iter=710, weight=[2.22958922], bias=[9.05306308], cost=[0.11411829]
iter=720, weight=[2.22150163], bias=[9.06733297], cost=[0.11143292]
iter=730, weight=[2.21373131], bias=[9.08104307], cost=[0.10895411]
iter=740, weight=[2.20626581], bias=[9.09421534], cost=[0.10666596]
iter=750, weight=[2.19909317], bias=[9.10687087], cost=[0.10455381]
iter=760, weight=[2.1922019], bias=[9.11902995], cost=[0.10260413]
iter=770, weight=[2.18558097], bias=[9.13071204], cost=[0.10080441]
iter=780, weight=[2.17921977], bias=[9.14193586], cost=[0.09914313]
iter=790, weight=[2.17310811], bias=[9.15271938], cost=[0.09760962]
iter=800, weight=[2.1672362], bias=[9.16307988], cost=[0.09619408]
iter=810, weight=[2.16159464], bias=[9.17303395], cost=[0.09488741]
iter=820, weight=[2.15617439], bias=[9.18259754], cost=[0.09368125]
iter=830, weight=[2.15096677], bias=[9.19178595], cost=[0.09256787]
iter=840, weight=[2.14596344], bias=[9.20061392], cost=[0.09154013]
iter=850, weight=[2.14115638], bias=[9.20909558], cost=[0.09059144]
iter=860, weight=[2.1365379], bias=[9.21724451], cost=[0.08971572]
iter=870, weight=[2.1321006], bias=[9.22507378], cost=[0.08890736]
iter=880, weight=[2.12783736], bias=[9.2325959], cost=[0.08816118]
iter=890, weight=[2.12374137], bias=[9.23982295], cost=[0.08747239]
iter=900, weight=[2.11980605], bias=[9.24676649], cost=[0.08683658]
iter=910, weight=[2.11602512], bias=[9.25343764], cost=[0.08624968]
iter=920, weight=[2.1123925], bias=[9.25984709], cost=[0.08570793]
iter=930, weight=[2.10890239], bias=[9.26600511], cost=[0.08520784]
iter=940, weight=[2.10554919], bias=[9.27192156], cost=[0.08474622]
iter=950, weight=[2.10232753], bias=[9.27760591], cost=[0.08432011]
iter=960, weight=[2.09923225], bias=[9.28306727], cost=[0.08392677]
iter=970, weight=[2.0962584], bias=[9.28831439], cost=[0.08356369]
iter=980, weight=[2.09340121], bias=[9.29335567], cost=[0.08322853]
iter=990, weight=[2.0906561], bias=[9.29819919], cost=[0.08291916]"""

W
#array([2.0882777])

# printing final values.
print('Final W value: {}\nFinal b value: {}'.format(W, b))
print('Final Cost/MSE(L2 Loss) Value: {}'.format(costs[-1]))
"""
Final W value: [2.08701682]
Final b value: [9.3391637]
Final Cost/MSE(L2 Loss) Value: [0.08914557]"""

# Plotting Line Plot for Number of Iterations vs MSE
plt.plot(range(iterations),costs)
plt.xlabel('Number of Iterations')
plt.ylabel('Mean Squared Error')
plt.title('Cost vs Iterations Analysis')

"""
the graph shows the cost decreasing exponentially as the iterations increase


Now we shall use our model to predict values of y from random values of X and test it to actual values of Y"""


# Prediction

# Generate data points like before
X_test = 3*np.random.rand(10,1)
y_test = 9 + 2*X_test + np.random.rand(10,1)

# Show the data point
data_string = 'Data point: {}, {}'
print(data_string.format(X_test, y_test))
"""
Data point: [[1.35491091]
 [0.4958074 ]
 [1.61141332]
 [0.05317137]
 [2.41643003]
 [1.80974883]
 [2.02236587]
 [2.34910188]
 [2.42159433]
 [0.44833339]], [[11.80762948]
 [10.09547954]
 [12.89303174]
 [ 9.50679577]
 [14.79796345]
 [12.95635894]
 [13.86802732]
 [14.31783609]
 [14.1386147 ]
 [10.20724226]]"""
 
prediction = predict(X_test, W, b)
prediction
"""
  array([[12.13182592],
       [10.33777922],
       [12.66747418],
       [ 9.41343226],
       [14.34857264],
       [13.0816538 ],
       [13.52565723],
       [14.20797274],
       [14.35935713],
       [10.23864031]])"""

score = float(sum(np.round(prediction) == np.round(y_test)))/ float(len(y_test))
score
#0.8 accuracy
"""
here  np.round(prediction) == np.round(y_test) means if the rounded number is equal then it is 1 else it is 0
it is then summed up and then divided by the number of data points there are for the accuracy"""

plt.scatter(X_test, y_test, color="blue", label="original")
plt.plot(X_test, prediction, color="red", label="predicted")
""" shows graph with predicted and actual"""

# Let's talk about one star now!

#Generate one data point
X_test = 3*np.random.rand(1,1)
y_test = 9 + 2*X_test + np.random.rand(1,1)

# Show the data point
data_string = 'Data point: {}, {}'
print(data_string.format(X_test, y_test))
# Data point: [[1.12110696]], [[11.7192993]]

prediction = predict(X_test, W, b)
prediction
#array([[11.64357834]])

score = float(sum(np.round(prediction) == np.round(y_test)))/ float(len(y_test))
score
# 1.0

if prediction >= thresh:
    print('We\'ve got a big star..')
else:
    print('We\'ve got a cute lil star!')
    
#We've got a cute lil star!




