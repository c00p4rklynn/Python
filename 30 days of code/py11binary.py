#!/bin/python3

import math
import os
import random
import re
import sys
import string

def binc(n) :
    bi = ""
    while n != 0 :
        a = n%2
        if a == 0 :
            bi = bi + "0"
            #print(bi, "a")
            n = n/2
        else:
            bi += "1"
            #print(bi)
            n = n-1
            n = n/2
    
    print(bi)
        
    

if __name__ == '__main__':
    n = int(input().strip())
    binc(n)
