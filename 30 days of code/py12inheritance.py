class Person:
	def __init__(self, firstName, lastName, idNumber):
		self.firstName = firstName
		self.lastName = lastName
		self.idNumber = idNumber
	def printPerson(self):
		print("Name:", self.lastName + ",", self.firstName)
		print("ID:", self.idNumber)

class Student(Person):
    #   Class Constructor
    #   Parameters:
    #   firstName - A string denoting the Person's first name.
    #   lastName - A string denoting the Person's last name.
    #   id - An integer denoting the Person's ID number.
    #   scores - An array of integers denoting the Person's test scores.
    # Write your constructor here
    def __init__(self, fname, lname, idd, scores) :
        Student.firstName = fname
        Student.lastName = lname
        Student.idNumber = idd
        Student.scores = scores
    #   Function Name: calculate
    def calculate(self) :
        s = len(Student.scores)
        i = 0
        su = 0
        for i in range(s) :
            su = su + Student.scores
        su = su/s
        print(su)
    #   Return: A character denoting the grade.
    #
    # Write your function here
    

line = input().split()
firstName = line[0]
lastName = line[1]
idNum = line[2]
numScores = int(input()) # not needed for Python
scores = list( map(int, input().split()) )
s = Student(firstName, lastName, idNum, scores)
s.printPerson()
print("Grade:", s.calculate())
