# June 22 2022
#notes questions and answers
# Control Flow

for i in range(5) :
  print(i)
# prints 0, 1, 2, 3, 4

# range(start, stop, step)

for index in range(0, 11, 2):
    print(index)
    
#

# break: used to terminate a for loop or a while loop prematurely.

for index in range(0, 10):
    print(index)
    if index == 3:
        break
#

# when used in a nested loop, break terminates the innermost loop
for x in range(5):
    for y in range(5):
        # terminates the innermost loop
        if y > 1:  
            break
        # shows coordinates on the screen
        print(f"({x},{y})")
#

# using break inside while

print('-- Help: type quit to exit --')
while True:
    color = input('Enter your favorite color:')
    if color.lower() == 'quit':
        break
#

# The continue statement skips the current iteration and starts the next one.

for index in range(10):
    if index % 2:
        continue
    print(index)
#

# continue inside while

index = 0
while index < 10:
    index += 1

    if not index % 2:
        continue

    print(index)
# prints 1 3 5 7 9

# The pass statement is a statement that does nothing. 
# It’s just a placeholder for the code that you’ll write in the future.

if index < 80:
    pass

for i in range(1,100):
    pass

while index < 100:
    pass

def fn():
    pass
  
#

#Functions

def read_file(f_to_read) :
  f = open(f_to_read)
  
  return f.read()
#

first_file_to_read = 'rocks.txt'

print(read_my_file(first_file_to_read))

second_file_to_read = 'text.txt'

print(read_my_file(second_file_to_read))

# python has no default params it is just null

def greet() :
  return 'hi'

greet()

# or

def greet(name):
  return f'Hi {name}'

greet(Alice)

#

def greet(name, message='Hey', question='Did you attend the last session?'):
    return f'{message} {name}, {question}'

greet('Rohan', 'Hello')

# you can change default value

# keyword arguments

def get_net_price(price, discount):
    return price * (1-discount) 
net_price = get_net_price(100, 0.1)
print(net_price)

net_price = get_net_price(discount=0.1, price=100)
print(net_price)

# recursive functions

# Since you need to stop counting down the number reaches zero, you add a condition like this:

def count_down(start):
    """ Count down from a number  """
    print(start)

    # call the count_down if the next
    # number is greater than 0
    next = start - 1
    if next > 0:
        count_down(next)


count_down(3)

"""Q1. Which of the following code would give [6, 8, 10] as its output?
A)

num = [i for i in range(0,11,2) if i >=  6]
print(num)
B)

num = [i for i in range(0,10,2) if i >=  6]
print(num)
C)

num = [if i >=  6 i for i in range(0,11,2)]
print(num)
D)

num = [if i >=  6 i for i in range(0,10,2)]
print(num)"""

# A is the answer:
num = [i for i in range(0,11,2) if i >=  6]
print(num)


"""Q2. What will be the output of the following code?

for letter in 'Spartificial':
  if 'i' in letter:
    pass
  print(letter , end = "")"""
#

# answer: Spartificial

"""Q3. What will the code below print?

for num_1 in range(7, -1, -1):
  for num_2 in range(2, num_1):
    if num_1 % num_2 == 1:
      print(num_1, end = " ")
      break
A) 7 6 5 4 3
B) 7 6 5 4 3 2 1 0 -1
C) 7 6 5 4 3 2 1 0
D) -1 0 1 2 3 4 5 6"""

# Answer is A

"""Q4. What will be the output of the code,

num1 = 6
num2 = 3

if (0 <= num1 <= 10) and (0 <= num2 <= 10):
  if num1 > num2:
    print(f"{num1} > {num2}")
  else:
    print(f"{num2} > {num1}")
A) True
B) False
C) 6
D) 6 > 3"""

# Answer is Line1: 55 ; Line2: This function only accepts a positive integer as an input! ; Line 3: same exception

"""Q6. Write a code to create a calculator which would calculate the sum of first n terms of Arithmetic Progression(AP) or Geometric Progression(GP)

User must be asked to choose if progression is AP or GP
If AP, ask for first term, common difference, number of terms
If GP, ask for first term, common ratio, number of terms
Calculate sum of first n terms from the information provided by the user
Calculator should only work if progression chosen is either AP or GP
Hint:- It will use a lot of if and else statements"""

# write your code here
prog_input = input("Enter AP for Arithmetic Progression or GP for Geometric Progression: ")

def AP(term1, diff, num) :
    a = 0
    for i in range(num) :
        a = a + term1 + (i)*diff
    print(a)
    return a

def GP(term1, rati, num) :
    a = 0
    for i in range(num) :
        a = a + term1*(rati**i)
    print(a)
    return


if prog_input == "AP" :
    f, n, t = [int(f) for f in input("Enter first term, common difference, number of terms with no commas inbetween: ").split()]
    AP(f, n, t)
    
elif prog_input == "GP" :
    f, n, t = [int(f) for f in input("Enter first term, common ratio, number of terms with no commas inbetween: ").split()]
    GP(f, n, t)
