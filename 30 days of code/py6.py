#!/bin/python3

import math
import os
import random
import re
import sys
import string



if __name__ == '__main__':
    n = int(input().strip())
    i = 1
    su = n
    for i in range(1, 11) :
        print(str(n) + " x " + str(i) + " = " + str(su))
        su = su + n
        
