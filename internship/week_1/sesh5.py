# June 24 2022
#python notes and questions and answers

#More Lists

colors = ['red', 'verde', 'purple']
p, c, b = colors

print(p)
print(c)
print(b)
# each variable stores one of the elements

p1, c1, *other = colors

print(p1)
print(c1)
print(other)
""" prints: purple
cyan
['black']"""

colors.append('blue')

p2, c2, *other = colors

print(p1)
print(c1)
print(other)

"""prints: 
purple
cyan
['black', 'blue']
*other is a pointer"""

for item in enumerate(flowers):
  print(item)

# this returns tuples

# what is we want the index and the flower separately?

for idx, flower in enumerate(flowers):
  print(f'{idx}: {flower}')
#

""" idx points to the index element of the tuple and flower to the element"""

flowers.index('ivy')
# helps find particular index of the particular element

# to find an element
to_find = 'hibiscus'

if to_find in flowers:
  print(flowers.index(to_find))
else:
  print(f'{to_find} is not found in the flowers list !')
#
# iteratables

# range function is an iterable

for idx in range(3):
    print(idx)
# every string is an iterable

my_favorite_language = 'my_favorite_language'

for ch in my_favorite_language:
  print(ch)
# now making a iter
flowers_iter = iter(flowers)

next_flower = next(flowers_iter)
print(next_flower)
# tulip
next_flower = next(flowers_iter)
print(next_flower)
# rose

colors = ['red', 'green', 'blue']
iterator = iter(colors)

for color in iterator:
    print(color)
#
# Lambda, Map, Reduce, Filter
"""Lambda

Lambda expressions are used to create anonymous functions - that have no names and are called just once.

They can have multiple arguments, but only one expression.

Usually look like: lambda parameters: expression"""

# Use Case 1 - functions accepting functions as arguments

def get_planet_info(planet_name, number_of_moons, no_of_named_moons, func_called):
    return func_called(planet_name, number_of_moons, no_of_named_moons)

def jupiter_info(planet_name, no_of_moons, no_of_named_moons):
  return f'{planet_name}: {no_of_moons} moons, {no_of_named_moons} named moons.'

def saturn_info(planet_name, no_of_moons, no_of_named_moons, no_of_rings=7):
  return f'{planet_name}: {no_of_moons} moons, {no_of_named_moons} named moons, {no_of_rings} rings.'

first_planet_info = get_planet_info('Jupiter', 79, 53, jupiter_info)
print(first_planet_info)
"""Jupiter: 79 moons, 53 named moons."""

second_planet_info = get_planet_info('Saturn', 82, 53, saturn_info)
print(second_planet_info)
"""Saturn: 82 moons, 53 named moons, 7 rings."""

new_first = get_planet_info('Jupiter', 79, 53, 
                            lambda planet_name, no_of_moons, no_of_named_moons: f'{planet_name}: {no_of_moons} moons, {no_of_named_moons} named moons.')

print(new_first)
"""Jupiter: 79 moons, 53 named moons."""

new_sec = get_planet_info('Saturn', 82, 53, 
                          lambda planet_name, no_of_moons, no_of_named_moons: f'{planet_name}: {no_of_moons} moons, {no_of_named_moons} named moons, 7 rings.')
print(new_sec)
"""Saturn: 82 moons, 53 named moons, 7 rings."""

# Use Case 2 - Functions returning functions

def times(n):
    return lambda x: x * n

doubler = times(2)
print(doubler)
""" prints the address of this function and thus doubler = lambda x: x*2"""

two_times_3 = doubler(3)
print(two_times_3)
print(doubler(5))
# prints 6 and 10

# MAP transformation of an entire list
# it calls a function on every item of a list and returns an iterator. maps one on one

my_marks = [80, 91, 78, 100, 87]

my_new_marks = []

for mark in my_marks:
    my_new_marks.append(mark*2)

print(my_new_marks)
"""[160, 182, 156, 200, 174]"""

my_new_marks_1 = map(lambda mark: mark*2, my_marks)
print(list(my_new_marks_1))
"""[160, 182, 156, 200, 174]"""

# applying map to strings

problems = ['finish College', 'Pay rent', 'study for EXAM']

big_problems = map(lambda problem:problem.upper(), problems)
print(list(big_problems))
""""['FINISH COLLEGE', 'PAY RENT', 'STUDY FOR EXAM']"""

small_prob = map(lambda prob:prob.lower(), problems)
print(list(small_prob))
"""['finish college', 'pay rent', 'study for exam']"""

# applying map to tuples

costs = [['Samsung', 400],
         ['Toshiba', 450],
         ['Lenovo', 700]]

tax = 0.1

costs = map(lambda item: [item[0], item[1], item[1] * tax], costs)

print(list(costs))

# costs = list(map(lambda item: [item[0], item[1], item[1] * tax], costs))
# print(costs)
# costs
"""[['Samsung', 400, 40.0], ['Toshiba', 450, 45.0], ['Lenovo', 700, 70.0]]"""



"""The filter() function iterates over the elements of the list and applies the fn() function to each element.

It returns an iterator for the elements where the fn() returns True.

filter(fn, list)"""

my_marks = [80, 51, 78, 60, 87]

good_marks = []

for mark in my_marks:
  if mark > 60:
    good_marks.append(mark)

print(good_marks)
"""[80, 78, 87]"""

good_marks_1 = list(filter(lambda mark: mark > 60, my_marks))

print(good_marks_1)
"""[80, 78, 87]"""

countries = [
    ['China', 1394015977],
    ['United States', 329877505],
    ['India', 1326093247],
    ['Indonesia', 267026366],
    ['Bangladesh', 162650853],
    ['Pakistan', 233500636],
    ['Nigeria', 214028302],
    ['Brazil', 21171597],
    ['Russia', 141722205],
    ['Mexico', 128649565]
]

populated = filter(lambda c: c[1] > 300000000, countries)
print(list(populated))
"""[['China', 1394015977], ['United States', 329877505], ['India', 1326093247]]"""



# REDUCE - to reduce a list of elements to one element.

my_marks = [80, 51, 78, 60, 87]

total = 0

for mark in my_marks:
    total += mark

print(total)

#356

# Unlike the map() and filter() functions, the reduce() isn’t a built-in function in Python. 
# It belongs to the functools module.
from functools import reduce

def sum(a, b):
  print(f"a = {a}, b = {b}, {a} + {b} = {a+b}")
  return a + b

total = reduce(sum, my_marks)
print(total)

"""a = 80, b = 51, 80 + 51 = 131
a = 131, b = 78, 131 + 78 = 209
a = 209, b = 60, 209 + 60 = 269
a = 269, b = 87, 269 + 87 = 356
356"""

"""The reduce() function cumulatively adds two elements of the list from left to right and reduces the whole list into a single value"""
total = reduce(lambda a, b: a + b, my_marks)

print(total)

"""Use list comprehensions instead of map() or filter() to make your code more concise and readable."""
my_marks = [80, 70, 100, 67, 89]

my_marks_squared = list(map(lambda mark:mark**2, my_marks))
print(my_marks_squared)
"""6400, 4900, 10000, 4489, 7921]"""

mmsq2 = [mark**2 for marks in my_marks]
print(mmsq2)
"""6400, 4900, 10000, 4489, 7921]"""


mountains = [
    ['Makalu', 8485],
    ['Lhotse', 8516],
    ['Kanchendzonga', 8586],
    ['K2', 8611],
    ['Everest', 8848]
]


highest_mountains = list(filter(lambda m: m[1] > 8600, mountains))
print(highest_mountains)
"""[['K2', 8611], ['Everest', 8848]]"""

hmount22 = [m for m in mountains if m[1] > 8600]
print(hmount22)
"""[['K2', 8611], ['Everest', 8848]]"""


# Dictionary Data Structure

empty_dict = {}

planet_info = {
    'name' : 'Jupiter',
    'number_of_moons' : 79,
    'number_in_solar_system' : 'fifth',
    'life_support' : 'some moons have oceans with water',
    'others' : 'gas_giant, biggest_planet, eternal_storms'
}

planet_info['number_of_moons']
"""79"""

# we can change values 

planet_info['life_support'] = 'Nothing concrete we know of so far'

# we cannot change keys

#planet_info['planet_name'] = 'Jupiter' we can only add keys

#del planet_info['planet_name']

moons = planet_info.get('number_of_moons')
# It returns the '0-00-000' string if the ssn key doesn’t exist in the dictionary:

storms = planet_info.get('storms', '0-00-000')

# looping through a dictionary

for key, value in planet_info.items():
    print(f"{key}: {value}")
#

for key in planet_info.keys():
    print(key)
for key in planet_info:
  print(key)
#

"""both print: 
name
number_of_moons
number_in_solar_system
life_support
others
planet_name"""

for value in planet_info.values():
    print(value)
# prints just the values


"""Dictionary Comprehension"""

stocks = {
    'AAPL': 121,
    'AMZN': 3380,
    'MSFT': 219,
    'BIIB': 280,
    'QDEL': 266,
    'LVGO': 144
}

new_stocks = {}
for symbol, price in stocks.items():
    new_stocks[symbol] = price*1.02

print(new_stocks)

new_stocks_1 = {symbol: price * 1.02 for (symbol, price) in stocks.items()}

print(new_stocks_1)

"""They both print: 
{'AAPL': 123.42, 'AMZN': 3447.6, 'MSFT': 223.38, 'BIIB': 285.6, 'QDEL': 271.32, 'LVGO': 146.88}"""

selected_stocks = {}
for symbol, price in stocks.items():
    if price > 200:
        selected_stocks[symbol] = price

print(selected_stocks)

#A dictionary comprehension iterates over items of a dictionary 
# and allows you to create a new dictionary by transforming or filtering each item.

selected_stocks_1 = {s: p for (s, p) in stocks.items() if p > 200}

print(selected_stocks_1)

"""prints: {'AMZN': 3380, 'MSFT': 219, 'BIIB': 280, 'QDEL': 266}"""


# Unpacking TUPLES

"""x, y = 10, 20, 30"""
""" When running the above code you will get an error saying that there are too many value to unpack you can use an extended unpacker which you can 
think of as a pointer using the symbol * """

r, g, *other = (192, 210, 100, 0.5)
print(other)
print(r)
print(g)

""" prints:
[100, 0.5]
192
210"""

""" x, y, *z, *t = (10, 20, 30, '10:30') does not work, python does not allow 2 unpacking operators"""

odd_numbers = (1, 3, 5)
even_numbers = (2, 4, 6)

# The following example uses the * operator to unpack those tuples and merge them into a single tuple:

numbers = (*odd_numbers, *even_numbers)
print(numbers)

"""(1, 3, 5, 2, 4, 6)"""

# *args

""" This can be used as a parameter to pass in multiple arguments into a functions eg:"""

def addition(*args):
  return sum(args)

print(addition(5, 10, 20, 6))
# 41

""" **kwargs can help us to pass in keyword arguments""""

def food(**kwargs):
  for items in kwargs:
    print(f"{kwargs[items]} is a {items}")
      
      
food(fruit = 'cherry', vegetable = 'potato', boy = 'srikrishna')
""" prints:
cherry is a fruit
potato is a vegetable
srikrishna is a boy"""



"""Time for some questions!

Q1. Which of the following statements are not true for lambda functions? Also, correct the statements

1. Doesn't require def or return to build a function
2. It's a short way of defining any normal functions in Python
3. Reduces lines of code to define a function
4. Call the function immediately after defining it at the end of defination
5. Often can be used inside another function such as map(), filter()
6. It can take any number of input arguments
7. It can have any number of expressions
8. Very difficult to construct as compared to normal def functions
9. We can use more than one line to construct them
10. Syntax of lambda function is:- "lambda arguments: expressions" 

answer:
1 - true
2 - true
3 - true
4 - true
5 - true
6 - true
7 - false
8 - kinda - depends on the person
9 - false
10 - true 



Q2. Print the output of tup1 in the following code,

tup = (1, 2, 3, 4, 5)
tup1 = tuple(map(lambda x: x/2, tup)) 

Answer: 
tup1 = (0.5, 1, 1.5, 2, 2.5)



Q3. Explain the code snippet posted below

def func(val, dict):
  for key, value in dict.items():
    if val == value:
      return key
  return "Key does not exist for the provided value!"
What will be the output of the following code?

moon_data = {
    "Mercury" : 0,
    "Venus" : 0,
    "Earth" : 1, 
    "Mars" : 2,
    "Jupiter" : 79,
    "Saturn" : 82, 
    "Uranus" : 27,
    "Neptune" : 14,
}

print(func(max(moon_data.values()), moon_data))      # Line 1
print(func(max(moon_data.values()) + 1, moon_data))  # Line 2

Answer: 
This is a functions definition which takes in a value and a dictionary and goes through the dictionary to find if there are any of the same value in the 
dictionary as given in the parameter and will return the key else it will return that the key does not exist

prints:
Saturn
Key does not exist for the provided value!




Q4. Remove empty strings from the list of physics concepts given below,*

phy_concepts = ["Gravitational Lensing", "", "Black Holes", "Quantum Entanglment", "", "Photoelectric Effect", "", "Gravitational Waves"]"""

# Given List
phy_concepts = ["Gravitational Lensing", "", "Black Holes", "Quantum Entanglment", "", "Photoelectric Effect", "", "Gravitational Waves"]

# Code for removing the empty strings 

phy2 = [i for i in phy_concepts if i != ""]

# Print the output
print(phy2)



"""
Q5. Add appropriate comments in the code such that anyone with basic knowledge of python would be able to understand what's going on here
"""


"""
prints original list

"""

lst = [1, 2, [3, 4, 5, [6], 7, 8, 9], 10, 12]

print(f"Original List:-  {lst}")

for i in list(range(2, 5)):

  if i == 2:
    for j in list(range(-3, 0)):
      lst[i][j] = lst[i][-j-1] # switches the position such that the list number increase and decrease
  
  else:
    lst[i] = lst[-i-1] # switching the position of the singular numbers of the list instead of the list in the list

print(f"Altered List:-   {lst}")

