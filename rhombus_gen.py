import matplotlib.pyplot as plt 
import numpy as np 
import string
import math
from PIL import Image

# PARAMETERS -- set the parameters of the image and the rhombus you want
w = 400 # width of image
h = 400 # height of image
length = 128 # side length fo rhombus
offset = 20 # offset of left side from the right side
line_width = 3 # thickness of line

total_height = length + offset
width = np.sqrt(length**2 - offset**2)
slope = offset/width # this is the slope of the two connecting sides

# initialize image
data = np.zeros((w, h), dtype=np.uint8) # start with blank image of size w, h
data[0:w-1, 0:h-1] = 255
img = Image.fromarray(data, 'L')
contrast_val = 255

# set right and left sides to black
for i in range(w): # y axis
	for j in range(h): # x axis
		if (j == 100 and (h + length)/2 >= i >= (h - length)/2): # takes care of left side
			data[i, j:j+line_width] = 255 - contrast_val
		elif (j == math.floor(100 + width) and (h + length)/2 >= i >= (h - length)/2): # takes care of right side which is shifted up 
			data[i - offset, j:j+line_width] = 255 - contrast_val

# we need a block just to calculate the actual coordinates for the edges
for j in range(100, math.floor(100 + width) + line_width): # all the x values for which this will be valid
	bottom_y = math.floor((j-100) * -1 * slope + (h + length) / 2) # bottom edge y values
	data[bottom_y:bottom_y+line_width, j] = 255 - contrast_val
	upper_y = bottom_y - length
	data[upper_y:upper_y+line_width, j] = 255 - contrast_val

# can also rotate ie 
# img.rotate(30).show()
img.show()
