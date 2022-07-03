# June 27 2022
# notes questions and answers

# CLASSES
""" classes are a way to combine information and behavior together like for example a rocket ship it has a position corresponding to x and y"""
class Rocket():
  
  def __init__(self) : # initalizing itself
    self.x = 0
    self.y = 0
    
""" first is the initialize itelf and set what the parameters which are needed to be defined its sefault or inital value; the self part allows you to access
the necessary variables outside the class when using it in the program"""

class Rocket():
    # Rocket simulates a rocket ship for a game,
    #  or a physics simulation.
    
    def __init__(self):
        # Each rocket has an (x,y) position.
        self.x = 0
        self.y = 0
        
    def move_up(self):
        # Increment the y-position of the rocket.
        self.y += 1
        
# Create a Rocket object.
my_rocket = Rocket()
print(my_rocket)
# prints: <__main__.Rocket object at 0x7fcd5581e950>

# here my_rocket is an object of class Rocket since we defined the varibale with a rocket class

class Rocket():
    # Rocket simulates a rocket ship for a game,
    #  or a physics simulation.
    
    def __init__(self):
        # Each rocket has an (x,y) position.
        self.x = 0
        self.y = 0
        
    def move_up(self):
        # Increment the y-position of the rocket.
        self.y += 1

# Create a Rocket object, and have it start to move up.
my_rocket = Rocket()
print("Rocket altitude:", my_rocket.y)
#Rocket altitude: 0

my_rocket.move_up()
print("Rocket altitude:", my_rocket.y)
#Rocket altitude: 1

my_rocket.move_up()
print("Rocket altitude:", my_rocket.y)
#Rocket altitude: 2

# you can create a fleet of rockets too

rockets = []
for i in range(5) :
  new_rocket = Rocket()
  rockets.append(new_rocket)
  
#you can also write it as:

myrock1 = [Rocket() for i in range(5)]
# move first rocket up
myrock1[0].move_up()

for rocket in myrock1 :
  print(f'Altitude: {rocket.y}')
""" prints:
Altitude: 1
Altitude: 0
Altitude: 0
Altitude: 0
Altitude: 0"""




# Object-Oriented terminology

"""Classes are part of the object oriented programming paradigm. OOP focuses on building reusable code such as classes; since we make use of an object
from a class it is called OOP.

A class is a body of code which defines the attributes and the behaviors required to accurately model a program. 

Rocket class:

__init__ method:
it is a special function in python; this method helps set all relevant variables or attributes to the default or initial values while self 
refers to the current object created from the class.


simple method of definition:
it is a function that is part of the class and you can do anything with it and self gives access to the calling objects attribute move_up is a simple method 
definition and when it is called it increments y by 1

init method automatically runs one time when it is called when you create a new object. 
What if you want to take in multiple parameters like different x and y coordinates? """

class Rocket():
    # Rocket simulates a rocket ship for a game,
    #  or a physics simulation.
    
    def __init__(self):
        # Each rocket has an (x,y) position.
        self.x = 0
        self.y = 0
        
    def move_up(self):
        # Increment the y-position of the rocket.
        self.y += 1
        
# we can change init to:



class Rocket() :
  
  def __init__(self, x=0, y=0) :
    self.x = x
    self.y = y
    
  def move_up(self):
    self.y += 1
#
#thus now init can accept new parameters to change the inital position of the rocket when creating a new object


rockets = []
rockets.append(Rocket())
rockets.append(Rocket(0,10))
rockets.append(Rocket(100,0))

# Show where each rocket is.
for index, rocket in enumerate(rockets):
    print("Rocket %d is at (%d, %d)." % (index, rocket.x, rocket.y))
#
""" prints:
Rocket 0 is at (0, 0).
Rocket 1 is at (0, 10).
Rocket 2 is at (100, 0)."""

""" we can also change the move_rocket function to allow us to move the rocket left and right on x axis and change the altitude more than one too"""

class Rocket():
    # Rocket simulates a rocket ship for a game,
    #  or a physics simulation.
    
    def __init__(self, x=0, y=0):
        # Each rocket has an (x,y) position.
        self.x = x
        self.y = y
        
    def move_rocket(self, x_increment=0, y_increment=1):
        # Move the rocket according to the paremeters given.
        #  Default behavior is to move the rocket up one unit.
        self.x += x_increment
        self.y += y_increment
        
# Create three rockets.
rockets = [Rocket() for x in range(0,3)]

# Move each rocket a different amount.
rockets[0].move_rocket()
rockets[1].move_rocket(10,10)
rockets[2].move_rocket(-10,0)
          
# Show where each rocket is.
for index, rocket in enumerate(rockets):
    print("Rocket %d is at (%d, %d)." % (index, rocket.x, rocket.y))
    
""" prints:
Rocket 0 is at (0, 1).
Rocket 1 is at (10, 10).
Rocket 2 is at (-10, 0)."""

""" what if we want to calculate the distance between 2 rockets for the pilot such that he knows how far away it is to pilot safely"""
from math import sqrt

class Rocket():
    # Rocket simulates a rocket ship for a game,
    #  or a physics simulation.
    
    def __init__(self, x=0, y=0):
        # Each rocket has an (x,y) position.
        self.x = x
        self.y = y
        
    def move_rocket(self, x_increment=0, y_increment=1):
        # Move the rocket according to the paremeters given.
        #  Default behavior is to move the rocket up one unit.
        self.x += x_increment
        self.y += y_increment
        
    def get_distance(self, other_rocket) :
      dist = sqrt((other_rocket.x - self.x)**2 - (other_rocket.y - self.x)**2)
      return dist
    

# 2 rockets
rock1 = Rocket()
rock2 = Rocket(10, 13)

# calculate distance

dist = rock1.get_distance(rock2)
print(f'The distance between rocket 1 and rocket 2 is {dist} meters away')

"""Prints: The distance between rocket 1 and rocket 2 is 16.401219 meters away"""




#INHERITANCE

"""it is one of the goals of OOP which is the creation of a stable reusable and reliable code. If you had to create a new class for each object then
that is hardly a reusable code. 

Inheritance is where you can inherit parts of an old class for a new class meaning basing the new class on an old class instead of creating a whole new class
The new class can override on unnecessary behaviors and create new attributes which are not in the parent class-also known as super class- for the child class
- also known as sub class-"""

#Spaceshuttle class

from math import sqrt

class Rocket():
    # Rocket simulates a rocket ship for a game,
    #  or a physics simulation.
    
    def __init__(self, x=0, y=0):
        # Each rocket has an (x,y) position.
        self.x = x
        self.y = y
        
    def move_rocket(self, x_increment=0, y_increment=1):
        # Move the rocket according to the paremeters given.
        #  Default behavior is to move the rocket up one unit.
        self.x += x_increment
        self.y += y_increment
        
    def get_distance(self, other_rocket):
        # Calculates the distance from this rocket to another rocket,
        #  and returns that value.
        distance = sqrt((self.x-other_rocket.x)**2+(self.y-other_rocket.y)**2)
        return distance
    
class Shuttle(Rocket):
    # Shuttle simulates a space shuttle, which is really
    #  just a reusable rocket.
    
    def __init__(self, x=0, y=0, flights_completed=0):
        super().__init__(x, y)
        self.flights_completed = flights_completed
        
shuttle = Shuttle(10,0,3)
print(shuttle)

""" now to define a subclass of the super class:

class newclass(parentclass):

and then for initialize the new class

  def __init__(self,argument_of_new_class, arguments_of_parent) :
    super().__init__(argument_of_parent) 

The super() function passes the self argument to the parent class automatically. You could also do this by explicitly naming the parent class 
when you call the __init__() function, but you then have to include the self argument manually:

class Shuttle(Rocket):
    # Shuttle simulates a space shuttle, which is really
    #  just a reusable rocket.
    
    def __init__(self, x=0, y=0, flights_completed=0):
        Rocket.__init__(self, x, y)
        self.flights_completed = flights_completed
"""

#Questions

from math import sqrt
from random import radint

class Rocket():
  
  def __init__(self, x=0, y=0):
    self.x = x
    self.y = y
    
  def move_rocket(self, x=0, y=1):
    self.x += x
    self.y += y
    
  def get_dist(self, other_rock):
    dist = sqrt((other_rock.x - self.x))** - (other_rock.y - self.y)**2)
    return dist
  
  
class shuttle(Rocket):
  
  def __init__(self, x=0, y=0, flights_completed=0):
    super().__init__(x, y)
    self.flights_completed = flights_completed
    
    
# create several shuttled and rockets with random positions and shuttles have random number of flights completed

shuttles = []
for i in range(0, 3):
  x = radint(0, 100)
  y = (1, 100)
  flights = radint(0, 10)
  shuttles.append(shuttle(x, y, flights))
  
rocks = []
for i in range(0, 3):
  x = radint(0, 100)
  y = radint(1, 100)
  rocks.append(Rocket(x, y))
  
#show the number of flights completed by the shuttle

for index, shut in enumerate(shuttles):
  print(f'Shuttle {index} has completed {shut.flights_completed} flight missions.')
  
print("\n")

# Show the distance from the first shuttle to all other shuttles.

for i in range(1, 3):
  num = i + 1
  dist = rocks[0].get_dist(rocks[i])
  print(f'The distance between shuttle 1 and shuttle {num} is {dist} units')
  
  
               
