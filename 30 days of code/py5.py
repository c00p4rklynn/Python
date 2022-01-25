class Person:
    def __init__(self,initialAge):
        
    def amIOld(self):
        
    def yearPasses(self):
       

t = int(input())
for i in range(0, t):
    age = int(input())         
    p = Person(age)  
    p.amIOld()
    for j in range(0, 3):
        p.yearPasses()       
    p.amIOld()
    print("")
