# June 29 2022
# notes and questions and answers

"""
SciPy
SciPy is a scientific computation library that uses NumPy underneath.

It provides more utility functions for optimization, stats and signal processing."""

from scipy import constants

# How many cubic meters are in one liter:
print(constants.liter)
#0.001

print(constants.pi)
#3.1415926535897935

print(dir(constants))
"""
['Avogadro', 'Boltzmann', 'Btu', 'Btu_IT', 'Btu_th', 'ConstantWarning', 'G', 'Julian_year', 'N_A', 'Planck', 'R', 'Rydberg', 'Stefan_Boltzmann', 
'Wien', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 
'_obsolete_constants', 'acre', 'alpha', 'angstrom', 'arcmin', 'arcminute', 'arcsec', 'arcsecond', 'astronomical_unit', 'atm', 'atmosphere', 
'atomic_mass', 'atto', 'au', 'bar', 'barrel', 'bbl', 'blob', 'c', 'calorie', 'calorie_IT', 'calorie_th', 'carat', 'centi', 'codata', 'constants', 
'convert_temperature', 'day', 'deci', 'degree', 'degree_Fahrenheit', 'deka', 'dyn', 'dyne', 'e', 'eV', 'electron_mass', 'electron_volt', 
'elementary_charge', 'epsilon_0', 'erg', 'exa', 'exbi', 'femto', 'fermi', 'find', 'fine_structure', 'fluid_ounce', 'fluid_ounce_US', 'fluid_ounce_imp', 
'foot', 'g', 'gallon', 'gallon_US', 'gallon_imp', 'gas_constant', 'gibi', 'giga', 'golden', 'golden_ratio', 'grain', 'gram', 'gravitational_constant', 'h', 
'hbar', 'hectare', 'hecto', 'horsepower', 'hour', 'hp', 'inch', 'k', 'kgf', 'kibi', 'kilo', 'kilogram_force', 'kmh', 'knot', 'lambda2nu', 'lb', 
'lbf', 'light_year', 'liter', 'litre', 'long_ton', 'm_e', 'm_n', 'm_p', 'm_u', 'mach', 'mebi', 'mega', 'metric_ton', 'micro', 'micron', 'mil', 
'mile', 'milli', 'minute', 'mmHg', 'mph', 'mu_0', 'nano', 'nautical_mile', 'neutron_mass', 'nu2lambda', 'ounce', 'oz', 'parsec', 'pebi', 'peta', 
'physical_constants', 'pi', 'pico', 'point', 'pound', 'pound_force', 'precision', 'proton_mass', 'psi', 'pt', 'short_ton', 'sigma', 'slinch',
'slug', 'speed_of_light', 'speed_of_sound', 'stone', 'survey_foot', 'survey_mile', 'tebi', 'tera', 'test', 'ton_TNT', 'torr', 'troy_ounce', 
'troy_pound', 'u', 'unit', 'value', 'week', 'yard', 'year', 'yobi', 'yotta', 'zebi', 'zepto', 'zero_Celsius', 'zetta']
"""

# Return the specified unit in meter (e.g. centi returns 0.01)
print(constants.yotta)    #1e+24
print(constants.zetta)    #1e+21
print(constants.exa)      #1e+18
print(constants.peta)     #1000000000000000.0
print(constants.tera)     #1000000000000.0
print(constants.giga)     #1000000000.0
print(constants.mega)     #1000000.0
print(constants.kilo)     #1000.0
print(constants.hecto)    #100.0
print(constants.deka)     #10.0
print(constants.deci)     #0.1
print(constants.centi)    #0.01
print(constants.milli)    #0.001
print(constants.micro)    #1e-06
print(constants.nano)     #1e-09
print(constants.pico)     #1e-12
print(constants.femto)    #1e-15
print(constants.atto)     #1e-18
print(constants.zepto)    #1e-21

# Return the specified unit in bytes (e.g. kibi returns 1024)
print(constants.kibi)    #1024
print(constants.mebi)    #1048576
print(constants.gibi)    #1073741824
print(constants.tebi)    #1099511627776
print(constants.pebi)    #1125899906842624
print(constants.exbi)    #1152921504606846976
print(constants.zebi)    #1180591620717411303424
print(constants.yobi)    #1208925819614629174706176


from scipy import linalg
import numpy as np
  
# The function takes two arrays
a = np.array([[7, 2], [4, 5]])
b = np.array([8, 10])

# Solving the linear equations
# remember that it works as ax = b thus: 7x + 2y = 8 and 4x + 5y = 10
res = linalg.solve(a, b)
print(res)
#[0.74074074 1.40740741]

# Initializing the matrix
x = np.array([[7, 2], [4, 5]])
  
# Finding the inverse of
# matrix x
y = linalg.inv(x)
print(y)
"""
[[ 0.18518519 -0.07407407]
 [-0.14814815  0.25925926]]"""

# Initializing the matrix
x = np.array([[8 , 2] , [3 , 5] , [1 , 3]])
  
# finding the pseudo inverse of matrix x
y = linalg.pinv(x)
print(y)
"""
[[ 0.14251208 -0.03381643 -0.03864734]
 [-0.07487923  0.16183575  0.11352657]]"""

# Initializing the matrix A
A = np.array([[9 , 6] , [4 , 5]])
  
# Finding the determinant of matrix A
D = linalg.det(A)
print(D)
#21.0

# The Singular-Value Decomposition is a matrix decomposition method for reducing a matrix to its constituent parts to make specific subsequent matrix calculations simpler.

# Initializing the matrix M
M = np.array([[1 , 5] , [6 , 10]]) 
  
# Passing the values to the 
# eigen function
x , y , z = linalg.svd(M)
print(x)
print(y)
print(z)
"""
[[-0.38684104 -0.92214641]
 [-0.92214641  0.38684104]]
[12.62901571  1.58365469]
[[-0.46873958 -0.88333641]
 [ 0.88333641 -0.46873958]]"""

# Initializing the matrix M
M = np.array([[9 , 3] , [2 , 4]])
  
# Passing the values to the eigen
# function
val , vect = linalg.eig(M)
  
# Display the Eigen values and Eigen
# vectors
print(val)
print(vect)
"""
[10.+0.j  3.+0.j]
[[ 0.9486833  -0.4472136 ]
 [ 0.31622777  0.89442719]]"""

# Initializing the input array 
x = np.array([6 , 3])
  
# Calculating the L2 norm
a = linalg.norm(x)
  
# Calculating the L1 norm
b = linalg.norm(x , 1)
  
# Displaying the norm values
print(a)
print(b)
"""
6.708203932499369
9.0"""

# Initializing the matrix 
x = np.array([[16 , 4] , [100 , 25]])
  
# Calculate and print the matrix 
# square root
r = linalg.sqrtm(x)
print(r)
print("\n")
  
# Calculate and print the matrix 
# exponential
e = linalg.expm(x)
print(e)
print("\n")
  
# Calculate and print the matrix
# sine
s = linalg.sinm(x)
print(s)
print("\n")
  
# Calculate and print the matrix 
# cosine
c = linalg.cosm(x)
print(c)
print("\n")
  
# Calculate and print the matrix 
# tangent
t = linalg.tanm(x)
print(t)
"""
[[ 2.49878019  0.62469505]
 [15.61737619  3.90434405]]


[[2.49695022e+17 6.24237555e+16]
 [1.56059389e+18 3.90148472e+17]]


[[-0.06190153 -0.01547538]
 [-0.38688456 -0.09672114]]


[[ 0.22445296 -0.19388676]
 [-4.84716897 -0.21179224]]


[[0.0626953  0.01567382]
 [0.39184561 0.0979614 ]]"""



# PANDAS 
#helps to use csv dataset files and create dataframes


import pandas as pd 

data = {
    'apples': [3, 2, 0, 1], 
    'oranges': [0, 3, 7, 2]
}

purchases = pd.DataFrame(data)

purchases
"""
	apples	oranges
0 	3	       0
1	  2	       3
2	  0	       7
3	  1	       2
pandas automatically creates a table and thus cant be shown in git hub
"""

purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])

purchases
"""
	    apples	oranges
June	   3	     0
Robert	 2	     3
Lily	   0	     7
David	   1	     2
index helps create an index for the table
"""

purchases.loc['June']
"""
helps locate what it is by index
apples     3
oranges    0
Name: June, dtype: int64"""

purchases.loc['Robert'].apples #prints out for the specific of apples header
# 2

purchases.apples # prints out for apples
"""
June      3
Robert    2
Lily      0
David     1"""

import pandas as pd
df = pd.read_csv('../input/ecommerce-purchases-csv/Ecommerce Purchases.csv')

df
"""
This opens a csv data set on ecommerce purchases with 10000 enteries and 14 information columns"""

df.head(5)
"""
prints first 5 enteries
"""

df.tail(5)
"""
it is the last 5 enteries
"""

df.info()
"""
it gives the information about the csv file such has the headers of each column and how many enteries are in each column and the data types
and also the memory usage of the file and also gives the range index or number of enteries eg:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 14 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Address           10000 non-null  object 
 1   Lot               10000 non-null  object 
 2   AM or PM          10000 non-null  object 
 3   Browser Info      10000 non-null  object 
 4   Company           10000 non-null  object 
 5   Credit Card       10000 non-null  int64  
 6   CC Exp Date       10000 non-null  object 
 7   CC Security Code  10000 non-null  int64  
 8   CC Provider       10000 non-null  object 
 9   Email             10000 non-null  object 
 10  Job               10000 non-null  object 
 11  IP Address        10000 non-null  object 
 12  Language          10000 non-null  object 
 13  Purchase Price    10000 non-null  float64
dtypes: float64(1), int64(2), object(11)
memory usage: 1.1+ MB"""

df.shape
# gives the shape of the table

df.drop_duplicates(inplace=True)
"""
Another important argument for drop_duplicates() is keep, which has three possible options:

first: (default) Drop duplicates except for the first occurrence.

last: Drop duplicates except for the last occurrence.

False: Drop all duplicates.

Syntax: DataFrame.drop_duplicates(subset=None, keep=’first’, inplace=False)
Parameters: 
subset: Subset takes a column or list of column label. It’s default value is none. After passing columns, it will consider them only for duplicates. 
keep: keep is to control how to consider duplicate value. It has only three distinct value and default is ‘first’. 


If ‘first’, it considers first value as unique and rest of the same values as duplicate.
If ‘last’, it considers last value as unique and rest of the same values as duplicate.
If False, it consider all of the same values as duplicates
inplace: Boolean values, removes rows with duplicates if True.
Return type: DataFrame with removed duplicate rows depending on Arguments passed. 
"""

temp_df = df.append(df)
print(temp_df.shape)
temp_df.drop_duplicates(inplace=True, keep=False)
print(temp_df.shape)
print(temp_df)

"""
here we make a new dataframe which is equal to df with another df appended to it. When you print the shape it is 20000 by 14. When you do drop duplicates 
inplace means remove rows with duplicates and it is true while keep is false thus it deletes all duplicates thus temp_df.shape is now 0 by 14."""

temp_df = df.append(df)
print(temp_df.shape)
temp_df.drop_duplicates(inplace=True, keep=first)
print(temp_df.shape)
print(temp_df)

"""
now only a flag in drop duplicate is changed where it keeps the first occurence of the duplicate while deletes all others. Thus the shape is now 10000 by 14
"""

df.columns
"""
Index(['Address', 'Lot', 'AM or PM', 'Browser Info', 'Company', 'Credit Card',
       'CC Exp Date', 'CC Security Code', 'CC Provider', 'Email', 'Job',
       'IP Address', 'Language', 'Purchase Price'],
      dtype='object')
"""

df.rename(columns={
        'CC Exp Date': 'cc-exp-date', 
        'CC Security Code': 'cc-security-code'
    }, inplace=True)
"""
Index(['Address', 'Lot', 'AM or PM', 'Browser Info', 'Company', 'Credit Card',
       'cc-exp-date', 'cc-security-code', 'CC Provider', 'Email', 'Job',
       'IP Address', 'Language', 'Purchase Price'],
      dtype='object')
"""

df.columns = [col.lower() for col in df]
""" makes all of the columns lowercase """

df.isnull()
# replaces everything to be null also known as false

df.isnull().sum()
"""
address             0
lot                 0
am or pm            0
browser info        0
company             0
credit card         0
cc-exp-date         0
cc-security-code    0
cc provider         0
email               0
job                 0
ip address          0
language            0
purchase price      0
dtype: int64
"""

df.dropna()
#drops all rows with NA or false in it 

dff.describe()
#describes the dta frame

df['am or pm'].describe()
"""
count     10000
unique        2
top          PM
freq       5068
Name: am or pm, dtype: object

gives information on one of the column headers"""

df['am or pm'].value_counts()
"""
gives proper value count
PM    5068
AM    4932
Name: am or pm, dtype: int64"""

df['company'].value_counts().head(10)
"""
Brown Ltd         15
Smith Group       15
Smith PLC         13
Smith LLC         13
Williams LLC      12
Smith and Sons    11
Davis and Sons    11
Brown Group       10
Johnson LLC       10
Johnson Ltd       10
Name: company, dtype: int64

value counts for first ten from the top"""

df.corr()
"""
it is used to find the pairwise correlation of all columns in the dataframe. Any na values are automatically excluded. For any non-numeric data type 
columns in the dataframe it is ignored."""

import seaborn as sns

sns.heatmap(df.corr(), cmap='viridis')
"""
prints out visualisation of df.corr"""

subset = df[['company', 'cc-security-code']]

subset.head()
"""
takes a subset of the dataframe. Thus subset has all the values of company and cc-security-code"""

df.loc[0]
"""
find locate information on the first value"""

print(df.loc[2, 'address'])
print('************************************************')
df.iloc[2,0]
"""
it gives the specific address of the number 2 in the dataframe and these are 2 ways to access it"""

df[df.iloc[:, 2] == 'PM']
""" help locate all the values which have PM in the am or pm column header"""

new = df["company"].isin(["Silva-Anderson", 'Wells Ltd'])
df[new]
""" find specific companies in company header and store in new and then search for it in df"""

filter1 = df["company"].isin(["Silva-Anderson", 'Wells Ltd'])
filter2 = df['am or pm'] == 'PM'

df[filter1 & filter2]
"""
make filters to filter out and get specific result like in this case get a compnay with the name silvia anderson or wels ltd while in 
am or pm colum it has to be PM in the column and find it in df"""

def filter_price(x):
  if x > 20.0:
    return 'good'
  else:
    return 'bad'
  
good_prices = df['purchase price'].apply(filter_price) 
good_prices[filter2].count()

"""
this helps filter things in the dataframe where x is more than 20 and is pm"""

"""
QUESTIONS AND ANSWERS:

Q-1 Write a program to

A. solve the following equation:

3x + 4y = 12

7x + 2y = 10

B. find determinant formed by the left side of both above equations.

C. find the eigen vector and eigen value of the matrix formed by the left side of both the equation in part A.

D. find the sin, cos and tan of the matrix formed by the left side of both the equation in part A."""

#A
a = numpy.array([[3, 4] , [7, 2]])
b = numpy.array([12, 10])
res = linalg.solve(a, b)

#B
D = linalg.det(a)

#C
val, vec = linalg.eig(a)

#D
si = linalg.sinm(a)
co = linalg.cosm(a)
ta = linalg.tanm(a)



"""
Q2. Given below is the Cars dataset, which contains all details about the cars included in the dataset. So, let's explore the dataset and perform following operations.

A. Read and explore the following dataset.

URL for the dataset : https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv

To get clear tabular view : https://github.com/selva86/datasets/blob/master/Cars93_miss.csv

B. Display top and last 15 rows of the dataset.

C. Figure out different Manufacturer present in the dataset and how many cars each manufacturer has in the market. Store the result in the form of dictionary, Manufacturer name as key and number of cars as value.

D. Find the total of cars having seating capacity more than 4.
518-452-8183
E. Drop all rows containing NaN values.

F. Drop columns named 'Turn.circle' and 'Weight'."""

# Read and Explore the dataset
import pandas as pd
carsdf = pd.read_csv('../input/cars93/cars93.csv')
carsdf

# Display top and last 15 rows of the dataset.
print(carsdf.head(15))
print(carsdf.tail(15))

# Figure out different Manufacturer present in the dataset and how many cars each manufacturer has in the market.
df = carsdf
print(df['Manufacturer'].value_counts())

# Find the total of cars having seating capacity more than 4.
fis = df['Passengers'] > 4
a = df[fis].count()
print(a)

# Drop all rows containing NaN values.
gf = df.dropna()
gf.info()


