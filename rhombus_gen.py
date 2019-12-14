import matplotlib.pyplot as plt 
import numpy as np 
import string
import math
from PIL import Image

# PARAMETERS -- set the parameters of the image and the rhombus you want
w = 500 # width of image
h = 500 # height of image
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
contrast_val = 128

# set right and left sides to black
for i in range(w): # y axis
	for j in range(h): # x axis
		if (j == 150 and (h + length)/2 >= i >= (h - length)/2): # takes care of left side
			data[i, j:j+line_width] = 255 - contrast_val
		elif (j == math.floor(150 + width) and (h + length)/2 >= i >= (h - length)/2): # takes care of right side which is shifted up 
			data[i - offset, j:j+line_width] = 255 - contrast_val

# we need a block just to calculate the actual coordinates for the edges
for j in range(150, math.floor(150 + width) + line_width): # all the x values for which the rhombus will be there
	bottom_y = int(round((j-150) * -1 * slope + (h + length) / 2)) # bottom edge y values
	# data[bottom_y:bottom_y+line_width, j] = 255 - contrast_val
	upper_y = bottom_y - length
	# data[upper_y:upper_y+line_width, j] = 255 - contrast_val
	data[upper_y:bottom_y+line_width, j] = 255 - contrast_val

# can also rotate if we want -- might be harder to get a good resolution though so let's hold off for now
# img = img.rotate(45)

translation = 20 # parameter to control how much it translates to the right

img_translate = img.copy()
img_cropped = img.crop((50, 50, w-50, w-50)) # crop to 400x400
img_cropped.show() # show orginal image cropped
img_translate = img.rotate(0, translate=(translation, 0)) # translates right by 20
img_translate_cropped = img_translate.crop((50, 50, w-50, w-50)) # crop to 400x400
img_translate_cropped.show() # show translated image cropped

data_translated = np.array(img_translate) # this gives us the translated image as np array

initial = data[50:w-50, 50:w-50] # this gives us the original image 400x400
final_translated = data_translated[50:w-50, 50:w-50] # this give us the translated image 400x400


