import numpy as np

#Area of Circle
r = float(input("Enter the radius of circle: "))
A = np.pi * r**2
print("Area of the circle is = " + str(A))

#Printing File Extension
filename = input("Enter a file name with extension: ")
f_extns = filename.split(".")
print("The extension of the file is: " + repr(f_extns[-1]))
