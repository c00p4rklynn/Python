import math
import os
import random
import re
import sys
import string

#1 Hello World
# Read a full line of input from stdin and save it to our dynamically typed variable, input_string.
input_string = input()

# Print a string literal saying "Hello, World." to stdout.
print('Hello, World.')
print(input_string)

# TODO: Write a line of code here that prints the contents of input_string to stdout.

#2 Data types
i = 4
d = 4.0
s = 'HackerRank '
# Declare second integer, double, and String variables. 
a = int(input())
b = float(input())
c = input()
# Read and save an integer, double, and String to your variables.
#a = a + i
print(a + i)
print(d + b)
print(s + c+'\n')
# Print the sum of both integer variables on a new line.

# Print the sum of the double variables on a new line.

# Concatenate and print the String variables on a new line
# The 's' variable above should be printed first.

#3 operators
def solve(meal_cost, tip_percent, tax_percent):
    # Write your code here
    a = round(meal_cost * (tip_percent/100))
    b = round(meal_cost * (tax_percent/100))
    print(int(meal_cost + a +b))

if __name__ == '__main__':
    meal_cost = float(input().strip())

    tip_percent = int(input().strip())

    tax_percent = int(input().strip())

    solve(meal_cost, tip_percent, tax_percent)
#

#4 conditional-statements
if __name__ == '__main__':
    N = int(input().strip())
    if N % 2 == 1 :
        print ("Weird")
    elif (N >= 2) & (N <= 5) :
        print ("Not Weird")
    elif (N >= 6) & (N <= 20) :
        print ("Weird")
    elif N > 20 :
        print ("Not Weird")
#

#
class Person:
    def __init__(self,initialAge):
        # Add some more code to run some checks on initialAge
        self.age = initialAge
        if self.age < 0 :
            self.age = 0
            print("Age is not valid, setting age to 0.")
    def amIOld(self):
        # Do some computations in here and print out the correct statement to the console
        if self.age < 13 :
            print("You are young.")
        elif self.age >= 13 and self.age < 18 :
            print("You are a teenager.")
        else :
            print("You are old.")
            
    def yearPasses(self):
        # Increment the age of the person in here
        self.age += 1

t = int(input())
for i in range(0, t):
    age = int(input())         
    p = Person(age)  
    p.amIOld()
    for j in range(0, 3):
        p.yearPasses()       
    p.amIOld()
    print("")
#


if __name__ == '__main__':
    n = int(input().strip())
    i = 1
    su = n
    for i in range(1, 11) :
        print(str(n) + " x " + str(i) + " = " + str(su))
        su = su + n
#

def seperator(word) :
    # function to separate the word into even and odd indexes
    i = 0
    s = len(word)
    ev = []
    od = []
    for i in range(s) :
        if i%2 == 0 :
            ev.append(word[i])
        else :
            od.append(word[i])
    a = len(ev)
    b = len(od)
    i = 0
    s = ""  
    for i in range (a):
        s += ev[i]
    i = 0
    s += " "
    for i in range (b):
        s += od[i]
    print(s)
    
if __name__ == '__main__' :
    t = 0
    t = int(input())
    i = 0
    for i in range(t) :
        wr = input()
        seperator(wr)
#



def factorial(n):
    # Write your code here
    fact = 0
    if n <= 1 :
        fact += 1
        return fact
    else :
        fact += n*factorial(n-1)
        
    return fact
            

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = factorial(n)

    fptr.write(str(result) + '\n')

    fptr.close()
#



