#June 21st 2022
#notes and question answers


number_ex_p = 5044 #number of exo planets
# format strings

print(f'There are {number_ex_p} exo planets')

print('There are {} exo planets'.format(number_ex_p))

# print the same thing by using format strings for variables with different values
number_of_planets = 8
x = f'There are {number_of_planets} number of planets in our solar system.'
print(x)

# Another way of using format strings

my_planet = 'Jupiter'
biggest_planet = 'This is the name of the biggest planet? {}'

print(biggest_planet.format(my_planet))

# use all CAPS in the variable name to remember it's a constant!

FILE_SIZE_LIMIT = 2000 

print(20 > 10)
print('z' > 'g')

# Getting the type of a value

type(100)

#input

value = input('Enter a value:')
print(value)

print("How old are you?", end = ' ') # end is used to specify how to end the printed content
age = input()
print(f'So you are {age} years old')

# Identity Operators

x = ["Artificial", "Intelligence"]
y = ["Artificial", "Intelligence"]
z = x

print(x is z)

# returns True because z is the same object as x

print(x is y)

# returns False because x is not the same object as y, even if they have the same content

print(x == y)

# to demonstrate the difference betweeen "is" and "==": this comparison returns True because x is equal to y


# Membership Operators

x = ["pear", "plum"]

print("pineapple" in x)

print("plum" in x)

print("pineapple" not in x)


# Bitwise Operators - for binary operations

a = 10

b = 4

print(a & b)
print(a | b)
print(~ a)

print(a ^ b)
print(a << b)
print(a >> b)

"""Q1. What will be the output for the following code,
a = input("Enter the value for a:- ")            
b = input("Enter the value for b:- ")               
print(a + b)
Assume that user will input a as 1 and b as 9
A) 10
B) 19
C) 1+9
D) Syntax Error
"""
# Answer is A

"""Q2. Write a code with membership operator to check if "Astrophysics" is present in the interest list mentioned below,
interest = ["Astronomy", "Astrophysics", "Deep Learning", "Football"]"""

if ("Astrophysics" in interest) :
  print("Astrophysics is present in the interest list")
else:
  print("Astrophysics is not present in the interest list")
  
#

"""Q3. We looked at % operator a while ago. What would a % b mean?
A) It divides a by b and converts the answer into percentage
B) It yields the reaminder from the division of a by b
C) It rounds off a upto b decimal places
D) It compares a with b as in which value is bigger"""

# Answer is B

"""Q4. Write a code using % and == operators to check if 2 and 3 are even and odd numbers respectively"""

for i in range(2):
  n = int(input("Enter in number: "))
  if (n%2==0) :
    print(f'The number {n} is an even number')
  elif (n%2==1) :
    print(f'The number {n} is odd')
    
#

a, b, c = 2, 4, 1                
D = b**2 - 4*a*c                   
print(f"For a = {a}, b = {b}, and c = {c}, \nDiscriminant = {D}")
