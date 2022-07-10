# June 28 2022
# notes and questions and answers

""" Use of NUMPY:
used for scientific computation"""

import numpy as np

# make array
my_array = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.int64)
print(my_array)
"""
[[1 2 3 4]
 [5 6 7 8]]"""

# Create an array of ones
a = np.ones((3,4))

# Create an array of zeros
b = np.zeros((2,3,4))

# Create an array with random values
c = np.random.random((2,2))

# Create an empty array
d = np.empty((3,2))

# Create a full array
e = np.full((2,2),7)

# Create an array of evenly-spaced values between the first num and second num while adding by the last num
f = np.arange(10,25,3)

# Create an array of evenly-spaced values
g = np.linspace(0,2,9)

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
"""
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
 
[[[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]

 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]]
  
[[0.29526218 0.07952625]
 [0.88842856 0.41896783]]
 
[[4.66805137e-310 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000]]
 
[[7 7]
 [7 7]]
 
[10 13 16 19 22]

[0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]"""

# Print the number of `my_array`'s dimensions
print(a.ndim)

# Print the number of `my_array`'s elements
print(a.size)

# Print information about `my_array`'s memory layout
#print(a.flags)

# Print the length of one array element in bytes
print(a.itemsize)

# Print the total consumed bytes by `my_array`'s elements
print(a.nbytes)

"""
2
12
8
96"""

# Print the length of `my_array`
print(len(a))

print(a)

# Change the data type of `my_array`
a.astype(float)

# Initialize `x`
x = np.ones((3,4))

# Check shape of `x`
print(x.shape)

# Initialize `y`
y = np.random.random((3,4))
print(y)

# Check shape of `y`
print(y.shape)

# Add `x` and `y`
x + y

"""
(3, 4)
[[0.61070301 0.02416972 0.64522847 0.39776354]
 [0.05273293 0.72027672 0.84049408 0.10242799]
 [0.00293826 0.95726776 0.01864118 0.35510066]]
(3, 4)

array([[1.61070301, 1.02416972, 1.64522847, 1.39776354],
       [1.05273293, 1.72027672, 1.84049408, 1.10242799],
       [1.00293826, 1.95726776, 1.01864118, 1.35510066]])"""


# Broadcasting works if one dimension of one of the numpy arrays is 1

# Initialize `x`
x = np.ones((3,4))

# Check shape of `x`
print(x.shape)

# Initialize `y`
y = np.arange(4)

# Check shape of `y`
print(y.shape)
print(y)

# Subtract `x` and `y`
x - y 

"""
(3, 4)
(4,)
[0 1 2 3]
array([[ 1.,  0., -1., -2.],
       [ 1.,  0., -1., -2.],
       [ 1.,  0., -1., -2.]])"""

# The shape of the resulting array will again be the maximum size along each dimension of x and y

# Initialize `x` and `y`
x = np.ones((3, 4))
print(x)

# y = np.random.random((5, 1, 4))
y = np.random.randint(1, 100, size=(5,1,4))
print(y)

# Add `x` and `y`
z = x + y
print(z.shape)
print(z)

"""
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
[[[48  7 13 38]]

 [[13 16 90 24]]

 [[83  9 55 79]]

 [[83 44 59 51]]

 [[ 3 27 92 53]]]
(5, 3, 4)
[[[49.  8. 14. 39.]
  [49.  8. 14. 39.]
  [49.  8. 14. 39.]]

 [[14. 17. 91. 25.]
  [14. 17. 91. 25.]
  [14. 17. 91. 25.]]

 [[84. 10. 56. 80.]
  [84. 10. 56. 80.]
  [84. 10. 56. 80.]]

 [[84. 45. 60. 52.]
  [84. 45. 60. 52.]
  [84. 45. 60. 52.]]

 [[ 4. 28. 93. 54.]
  [ 4. 28. 93. 54.]
  [ 4. 28. 93. 54.]]]"""

# Add `x` and `y`
print(x)
print(y)
sum = np.add(x,y)
print(sum)

# Subtract `x` and `y`
diff = np.subtract(x,y)
print(diff)

# Multiply `x` and `y`
prod = np.multiply(x,y)
print(prod)

# Divide `x` and `y`
quo = np.divide(x,y)
print(quo)

# Calculate the remainder of `x` and `y`
rem = np.remainder(x,y)
print(rem)

"""
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
[[[48  7 13 38]]

 [[13 16 90 24]]

 [[83  9 55 79]]

 [[83 44 59 51]]

 [[ 3 27 92 53]]]
[[[49.  8. 14. 39.]
  [49.  8. 14. 39.]
  [49.  8. 14. 39.]]

 [[14. 17. 91. 25.]
  [14. 17. 91. 25.]
  [14. 17. 91. 25.]]

 [[84. 10. 56. 80.]
  [84. 10. 56. 80.]
  [84. 10. 56. 80.]]

 [[84. 45. 60. 52.]
  [84. 45. 60. 52.]
  [84. 45. 60. 52.]]

 [[ 4. 28. 93. 54.]
  [ 4. 28. 93. 54.]
  [ 4. 28. 93. 54.]]]
[[[-47.  -6. -12. -37.]
  [-47.  -6. -12. -37.]
  [-47.  -6. -12. -37.]]

 [[-12. -15. -89. -23.]
  [-12. -15. -89. -23.]
  [-12. -15. -89. -23.]]

 [[-82.  -8. -54. -78.]
  [-82.  -8. -54. -78.]
  [-82.  -8. -54. -78.]]

 [[-82. -43. -58. -50.]
  [-82. -43. -58. -50.]
  [-82. -43. -58. -50.]]

 [[ -2. -26. -91. -52.]
  [ -2. -26. -91. -52.]
  [ -2. -26. -91. -52.]]]
[[[48.  7. 13. 38.]
  [48.  7. 13. 38.]
  [48.  7. 13. 38.]]

 [[13. 16. 90. 24.]
  [13. 16. 90. 24.]
  [13. 16. 90. 24.]]

 [[83.  9. 55. 79.]
  [83.  9. 55. 79.]
  [83.  9. 55. 79.]]

 [[83. 44. 59. 51.]
  [83. 44. 59. 51.]
  [83. 44. 59. 51.]]

 [[ 3. 27. 92. 53.]
  [ 3. 27. 92. 53.]
  [ 3. 27. 92. 53.]]]
[[[0.02083333 0.14285714 0.07692308 0.02631579]
  [0.02083333 0.14285714 0.07692308 0.02631579]
  [0.02083333 0.14285714 0.07692308 0.02631579]]

 [[0.07692308 0.0625     0.01111111 0.04166667]
  [0.07692308 0.0625     0.01111111 0.04166667]
  [0.07692308 0.0625     0.01111111 0.04166667]]

 [[0.01204819 0.11111111 0.01818182 0.01265823]
  [0.01204819 0.11111111 0.01818182 0.01265823]
  [0.01204819 0.11111111 0.01818182 0.01265823]]

 [[0.01204819 0.02272727 0.01694915 0.01960784]
  [0.01204819 0.02272727 0.01694915 0.01960784]
  [0.01204819 0.02272727 0.01694915 0.01960784]]

 [[0.33333333 0.03703704 0.01086957 0.01886792]
  [0.33333333 0.03703704 0.01086957 0.01886792]
  [0.33333333 0.03703704 0.01086957 0.01886792]]]
[[[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]

 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]

 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]

 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]

 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]]"""

print(x)
print(x.min())
print(x.sum())
print(x.max(axis=0)) # find the max along the axis where axis gives if it is the row or column or etc thus here 0 gives for row
print(x.cumsum(axis=1)) #cumulative sum on which axis
print(x.mean())
print(np.std(a))

"""
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
1.0
12.0
[1. 1. 1. 1.]
[[1. 2. 3. 4.]
 [1. 2. 3. 4.]
 [1. 2. 3. 4.]]
1.0
0.0"""

# `a` AND `b` 
print(a)
print(b)
c = np.logical_and(a, b)
print(c)
# `a` OR `b` 
c = np.logical_or(a, b)
print(c)
# `a` NOT `b` 
c = np.logical_not(a,b)
print(c)

"""
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
[[[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]

 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]]
[[[False False False False]
  [False False False False]
  [False False False False]]

 [[False False False False]
  [False False False False]
  [False False False False]]]
[[[ True  True  True  True]
  [ True  True  True  True]
  [ True  True  True  True]]

 [[ True  True  True  True]
  [ True  True  True  True]
  [ True  True  True  True]]]
[[[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]

 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]]"""

# Make the array `my_array`
my_2d_array = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int64)

# Print `my_array`
print(my_2d_array)
print(my_2d_array.ndim)
"""
[[1 2 3 4]
 [5 6 7 8]]
2"""

my_3d_array = np.random.randint(1, 100, size=(5,2,4))
print(my_3d_array.ndim)
print(my_3d_array)

"""
3
[[[20 35 26 56]
  [83 53 74 53]]

 [[82 79 97 97]
  [54 96 13 46]]

 [[75 75 95 46]
  [41 79 85  8]]

 [[15 27 89 20]
  [76 27 28 56]]

 [[39 69 97 38]
  [17 62 79 75]]]"""

# Select the element at row 1 column 2
print(my_2d_array[1][2])

# Select the element at row 1 column 2
print(my_2d_array[1,2])

# Select the element at row 1, column 2 and 
print(my_3d_array[1,1,2])
"""
7
7
29"""

# Select items at index 0 and 1
print(my_2d_array[0:2])

# Select items at row 0 and 1, column 1
print(my_2d_array[0:2,1])

# Select items at row 1
# This is the same as saying `my_3d_array[1,:,:]
print(my_3d_array[1,...])

"""
[1 2 3 4]
 [5 6 7 8]]
[2 6]
[[52 83 86 86]
 [73 15 29 15]]"""

#find things in a array depending on a condition
# Try out a simple example
print(my_2d_array[my_2d_array<2])

# Specify a condition
bigger_than_3 = (my_3d_array >= 3)

# Use the condition to index our 3d array
print(my_3d_array[bigger_than_3])
"""
[1]
[81 10 97 36 49 91 48 60 52 83 86 86 73 15 29 15 82  3 74 19 76  6 29 65
 82 48 50 71 65 36 40 92 13 16 89 83 26 81 76  5]
"""


print(y)
print(np.sort(y, axis = 0))
print(y.transpose())
print(y.T)
"""
[[[45 86 79  7]]

 [[19 27 65 37]]

 [[73 42 55  9]]

 [[51 73 44 99]]

 [[29 97 21 64]]]
[[[19 27 21  7]]

 [[29 42 44  9]]

 [[45 73 55 37]]

 [[51 86 65 64]]

 [[73 97 79 99]]]
[[[45 19 73 51 29]]

 [[86 27 42 73 97]]

 [[79 65 55 44 21]]

 [[ 7 37  9 99 64]]]
[[[45 19 73 51 29]]

 [[86 27 42 73 97]]

 [[79 65 55 44 21]]

 [[ 7 37  9 99 64]]]"""

first = np.array([1, 2, 3])
second = np.array([4, 5, 6])

import numpy.ma as ma

con = ma.concatenate([first, second])
print(con.data)

print(ma.concatenate([first, second]))
"""
[1 2 3 4 5 6]
[1 2 3 4 5 6]"""


square = np.array([
    [16, 3, 2, 13],
    [5, 10, 11, 8],
    [9, 6, 7, 12],
    [4, 15, 14, 1]
])

for i in range(4):
    assert square[:, i].sum() == 34
    print(square[:, i])
    print("nnnn")
    assert square[i, :].sum() == 34


assert square[:2, :2].sum() == 34

assert square[2:, :2].sum() == 34

assert square[:2, 2:].sum() == 34

assert square[2:, 2:].sum() == 34

"""
[16  5  9  4]
nnnn
[ 3 10  6 15]
nnnn
[ 2 11  7 14]
nnnn
[13  8 12  1]
nnnn
"""

square[2:, :2]
"""
array([[ 9,  6],
       [ 4, 15]])"""

numbers = np.linspace(5, 50, 24, dtype=int).reshape(4, 6) #reshape helps to reshape the array to a diffeent type

numbers
"""
array([[ 5,  6,  8, 10, 12, 14],
       [16, 18, 20, 22, 24, 26],
       [28, 30, 32, 34, 36, 38],
       [40, 42, 44, 46, 48, 50]])"""

mask = numbers % 4 == 0
mask
"""
array([[False, False,  True, False,  True, False],
       [ True, False,  True, False,  True, False],
       [ True, False,  True, False,  True, False],
       [ True, False,  True, False,  True, False]])"""

numbers[mask]
"""array([ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])"""
by_four = numbers[numbers % 4 == 0]



from numpy.random import default_rng #this is for a refault random nuber generator

rng = default_rng()

values = rng.standard_normal(10000) #has 10000 random values
values[:5]
"""array([ 0.93960932,  0.08244221, -0.69671395,  0.80155414, -0.26466507])"""

std = values.std()
std
"""0.9955307778481921"""

filtered = values[(values > -2 * std) & (values < 2 * std)] #helps filter values
filtered.size / values.size
"""0.9565"""

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr)
"""[array([1, 2]), array([3, 4]), array([5, 6])]"""

arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

newarr = np.array_split(arr, 3)

print(newarr[0])
print(newarr[1])
print(newarr[2])

"""
[[1 2]
 [3 4]]
[[5 6]
 [7 8]]
[[ 9 10]
 [11 12]]"""

# split along rows

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3, axis=1)
# newarr = np.array_split(arr, 3)

print(arr)
print(newarr[0])
print(newarr[1])
print(newarr[2])
"""
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]
 [13 14 15]
 [16 17 18]]
[[ 1]
 [ 4]
 [ 7]
 [10]
 [13]
 [16]]
[[ 2]
 [ 5]
 [ 8]
 [11]
 [14]
 [17]]
[[ 3]
 [ 6]
 [ 9]
 [12]
 [15]
 [18]]"""

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.hsplit(arr, 3)

print(newarr[0])
print(newarr[1])
print(newarr[2])
"""
[[ 1]
 [ 4]
 [ 7]
 [10]
 [13]
 [16]]
[[ 2]
 [ 5]
 [ 8]
 [11]
 [14]
 [17]]
[[ 3]
 [ 6]
 [ 9]
 [12]
 [15]
 [18]]"""


arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)

print(arr)
"""
[[1 2 5 6]
 [3 4 7 8]]"""

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)
"""
[[1 4]
 [2 5]
 [3 6]]"""

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.vstack((arr1, arr2))
print(arr)

arr = np.hstack((arr1, arr2))
print(arr)

arr = np.dstack((arr1, arr2))
print(arr)
"""
[[1 2 3]
 [4 5 6]]
 
 [1 2 3 4 5 6]
 
 [[[1 4]
  [2 5]
  [3 6]]]
 """

a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 
np.dot(a,b)

# [[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]
"""
[[37, 40]
 [85, 92]]"""

a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 
np.vdot(a,b)

# 1*11 + 2*12 + 3*13 + 4*14 = 130
"""130 """

a,b = 13,17 
print(bin(a))
print(bin(b))

print(np.bitwise_and(13, 17))
"""
0b1101
0b10001
1"""

np.bitwise_or(13, 17)
#29

np.invert(np.array([13], dtype = np.uint8)) 
np.binary_repr(13, width = 8) 
np.binary_repr(242, width = 8)

print('Left shift of 10 by two positions:' )
print(np.left_shift(10,2) )


print('Binary representation of 10:' )
print(np.binary_repr(10, width = 8) )
 

print('Binary representation of 40:' )
np.binary_repr(40, width = 8)  
"""
Left shift of 10 by two positions:
40
Binary representation of 10:
00001010
Binary representation of 40:
'00101000'"""

print('Left shift of 10 by two positions:' )
print(np.right_shift(40,2) )


print('Binary representation of 10:' )
print(np.binary_repr(40, width = 8) )
 

print('Binary representation of 40:' )
np.binary_repr(10, width = 8)  
"""
Left shift of 10 by two positions:
10
Binary representation of 10:
00101000
Binary representation of 40:
'00001010'
"""

print('Concatenate two strings:' )
print(np.char.add(['hello'],[' xyz']) )

print('Concatenation example:' )
print(np.char.add(['hello', 'hi'],[' abc', ' xyz']))
"""
Concatenate two strings:
['hello xyz']
Concatenation example:
['hello abc' 'hi xyz']"""

np.char.multiply('Hello ',3)
"""array('Hello Hello Hello ', dtype='<U18')"""

array('Hello Hello Hello ', dtype='<U18')
"""array('*******hello********', dtype='<U20')"""

array('*******hello********', dtype='<U20')
"""array('Hello world', dtype='<U11')"""

np.char.title('hello how are you?')
"""array('Hello How Are You?', dtype='<U18')"""

np.char.lower(['HELLO','WORLD']) 
"""array(['hello', 'world'], dtype='<U5')"""

np.char.upper('hello') 
"""array('HELLO', dtype='<U5')"""

print(np.char.split ('hello how are you?') )
print(np.char.split ('TutorialsPoint, Hyderabad, Telangana', sep = ','))
"""
['hello', 'how', 'are', 'you?']
['TutorialsPoint', ' Hyderabad', ' Telangana']"""

print(np.char.splitlines('hello\nhow are you?') )
print(np.char.splitlines('hello\rhow are you?'))

#This function returns a list of elements in the array, breaking at line boundaries.
"""['hello', 'how are you?']"""

print(np.char.strip('ashok arora','a') )
print(np.char.strip(['arora','admin','java'],'a'))

#This function returns a copy of array with elements stripped of the specified characters leading and/or trailing in it.
"""
shok aror
['ror' 'dmin' 'jav']"""


print(np.char.join(':','dmy') )
print(np.char.join([':','-'],['dmy','ymd']))
"""
d:m:y
['d:m:y' 'y-m-d']"""

np.char.replace ('He is a good boy', 'is', 'was')
"""array('He was a good boy', dtype='<U17')"""

#QUESTIONS AND ANSWERS
"""
Q1. Flatten the following array.

A. [[1,2,3], [4,5,6], [7,8,9]]
reshape to 1 dimension
"""
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(a)
l = a.size
a.reshape(l)
a.flatten() # this also flattens an array to one dimenstion

"""
Q2. What will the output of the following code and explain the code with proper comments.

import numpy as np

arr = []
temp = []

for i in range(12):
    temp.append(np.random.rand()*10)
    if((len(temp)) % 3 == 0):
        arr.append(temp)
        temp = []

arr =  np.array(arr)
print(arr.flatten())
print(arr)

In the for loop it will be looped 12 times where the temp array has three random values appended to it before it is then appended to arr and made 
empty again. Thus arr will be an array of 4 by 3 and arr.flatten flattens the array to one dimension

[7.72928536 7.62389378 3.05969389 5.1884725  6.51949687 2.82775201
 2.15779553 6.40335718 4.2729721  1.50329595 9.01982289 2.97444823]
[[7.72928536 7.62389378 3.05969389]
 [5.1884725  6.51949687 2.82775201]
 [2.15779553 6.40335718 4.2729721 ]
 [1.50329595 9.01982289 2.97444823]]"""

"""Q3. Explain the difference between the following and when they are used?

A. np.identity() and np.eye()

B. np.zeros() and np.ones()

C. np.random.randint() and np.random.rand()

A)
It returns a 2d identity matrix but eye you can set the diagonal numbers in the identity matrix.

B)
zeros return a matrix full of 0 while ones give 1s

c)
randint gives a matrix full of random integers while rand just gives random numbers



Q4. What will be the output of the following code?

import numpy as np

a = np.array( [[21,15, 19, 18, 17],[16, 20, 14, 13, 12], [11, 10 , 9 , 8 , 7], [6 , 5,  4,  3,  2]])
b = a[:,1]
b = sorted(b)
c = a.copy()
k = 0
for i in b:
    index = np.where(a[:,1] == i)
    c[k] = a[index]
    k+=1

print(c)


[[ 6  5  4  3  2]
 [11 10  9  8  7]
 [21 15 19 18 17]
 [16 20 14 13 12]]"""
