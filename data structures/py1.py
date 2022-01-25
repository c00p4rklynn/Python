#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'hourglassSum' function below.
#
# The function is expected to return an INTEGER.
# The function accepts 2D_INTEGER_ARRAY arr as parameter.
#

def hourglassSum(arr):
    # Write your code here
    i = 0
    max = 0
    for i in list(range(4)):
        j = 0
        for j in list(range(4)):
            sum = arr[i][j] + arr[i+1][j] + arr[i+2][j]
            print (sum)
            sum = arr[i+1][j+1]
            sum = arr[i][j+2] + arr[i+1][j+2] + arr[i+2][j+2]
            print (sum)
            if sum > max :
                max = sum
                
    print(max)
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr = []

    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    result = hourglassSum(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
