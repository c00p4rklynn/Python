# June 23 2022
#notes and questions and answers

# SETS
# This is a set you use quare brackets
languages = ["python", "Java", "Cpp", "php"]
print(languages)

# you can access elements backwards with negative index numbers
marks = [85, 75, 100, 80, 43, 13]
print(marks[-1])
# prints 13

# you can replace elements in the list too
marks[2] = 65
marks[-2] = 48

marks.append(31) # appends the number to the end
marks.insert(1, 97) #inserts number at the position of 1
del marks[3] # deletes number at position 3
print(marks.pop()) #removes last element while being able to use the number
marks.remove(80) #remves the specific number from the set

#IMMUTABLE SETS TUPLES

healthy_food_choices = ('eggs', 'quinoa', 'veggies')

print(healthy_food_choices[-1])
print(healthy_food_choices)

# sort function

marks = [80, 91, 78, 100, 87]
marks.sort()
#this sorts it to least to greatest
#same for words
marks.sort(reverse=True)
#this sorts from greatest to least

companies = [('Meta', 2021, 150.6),
             ('Apple', 2019, 102.8),
             ('Netflix', 2020, 95.6)]

companies.sort()
print(companies)

companies.sort(reverse=True)
print(companies) # default sorting happens as per 1st value of tuple

# what if you want to sort by a parameter set by you? 

def sort_key(company):
    return company[2]

companies.sort(key=sort_key)
print(companies)

companies.sort(key=sort_key, reverse=True)
print(companies)
#print: [('Netflix', 2020, 95.6), ('Apple', 2019, 102.8), ('Meta', 2021, 150.6)]
#print: [('Meta', 2021, 150.6), ('Apple', 2019, 102.8), ('Netflix', 2020, 95.6)]

# To return the new sorted list from the original list, you use the sorted() function:

# The original list does not change here!

my_students = ['Shubham', 'Atharv', 'Prasad', 'Ahaan', 'Selva', 'Prateek', 'Deena']
my_sorted_students = sorted(my_students)

print(my_students)
print(my_sorted_students)

my_students = ['Shubham', 'Atharv', 'Prasad', 'Ahaan', 'Selva', 'Prateek', 'Deena']
my_sorted_students_desc = sorted(my_students, reverse=True)

print(my_students)
print(my_sorted_students_desc)

# list[begin: end: step]

flowers = ['tulip', 'rose', 'lily', 'iris', 'daisy', 'orchid', 'jasmine', 'poppy', 'ivy', 'violet', 'holly', 'heather']

chosen_flowers = flowers[2:5]
print(chosen_flowers)

chosen_flowers_new = flowers[2:13:3]
print(chosen_flowers_new)

# To get the n-first elements from a list, you omit the begin argument: list[:n]

print(flowers[:5])

# this is actually equivalent to

print(flowers[0:5])

# To return a sublist that includes every nth element:

flowers[::2]

#print: ['tulip', 'lily', 'daisy', 'jasmine', 'ivy', 'holly']

# Using Python List slice to reverse a list

reversed_flowers = flowers[::-1]
print(reversed_flowers)

# When you use a negative step, the slice includes the list elements starting from the last element to the first element. 

reversed_flowers = flowers[::-2]
print(reversed_flowers)

#print: ['heather', 'holly', 'violet', 'ivy', 'poppy', 'jasmine', 'orchid', 'daisy', 'iris', 'lily', 'rose', 'tulip']
#print: ['heather', 'violet', 'poppy', 'orchid', 'iris', 'rose']

# Using Python List slice to substitute part a list

flowers[0:2] = ['dahlia', 'peony']
print(flowers)

# Using Python List slice to partially replace and resize a list

flowers[0:2] = ['hyacinth', 'lila', 'yolanda']
print(flowers)

flowers[0:2] = ['sage']
print(flowers)

# 0 to 2 flowers got replaced with sage

del flowers[5:7]
print(flowers)

#print: ['sage', 'yolanda', 'lily', 'iris', 'daisy', 'poppy', 'ivy', 'violet', 'holly', 'heather']

"""Q1. Write at max 2 lines of code (excluding lines used for commenting) to,

Crate a list of first ten positive integers
Cube only the odd numbers from that list
The output of the code should result into:
1^3 = 1
3^3 = 27
5^3 = 125
7^3 = 343
9^3 = 729"""

# Code here
num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num1 = num[::2]
#a = length(num1)
for i in num1 :
    print(str(i)+"^3 =",(i**3))
    
#
"""Q2. What will be the output of the code below,

nums = [[1, 2, 3, 4, 5, 6], 
        [7, 8, 9, 10, 11, 12], 
        [13, 14, 15, 16, 17, 18]]

for i in range(len(nums)):
  if i < 2:
    print(nums[i][::2])
  else:
    print(nums[i][::-2])"""

""" answer: [1, 3, 5]
[7, 9, 11]
[18, 16, 14]"""

"""Q3 Which of the following is an invalid list

A) [1,2,'abc',1+5j, 3.14]

B) [ [ (1,2),9 ] , ('m','n','o') ]

C) list("Student")

D) All are Valid."""

#D

"""Q4 What will be the output of the following.

arr = [[11, 12, 13, 14],
       [44, 55, 66, 87],
       [81, 91, 10, 111],
       [56, 16, 17, 89]]

for _ in range(3):
    print(arr[_].pop())"""
# 14 87 111

