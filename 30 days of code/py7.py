# Enter your code here. Read input from STDIN. Print output to STDOUT
import string

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
