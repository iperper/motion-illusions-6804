import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# need to initialize data we collected in the csv file -- make a csv file so that we can easily parse it
data = {} # dictionary where keys are a 3 tuple of (texture, angle, contrast) -- objects are an array of tuples of form (depth, id)
data[(0, 20, 0)] = [(16, 12), (60, 13), (10, 14), (2, 15), (8, 17), (5, 18)]
data[(0, 20, 200)] = [(1, 3), (5, 7), (3, 8), (.3, 10), (5, 18)]
data[(0, 20, 245)] = [(2, 4), (4, 10), (5, 11), (16, 12), (60, 13), (10, 14), (4, 18)]
data[(0, 40, 0)] = [(20, 5), (10, 7), (15, 9), (10, 11), (20, 12), (20, 14), (3, 15)]
data[(0, 40, 200)] = [(10, 2), (8, 3), (8, 6), (15, 7), (20, 9)]
data[(0, 40, 245)] = [(15, 2), (8, 3), (5, 4), (20, 5), (20, 9), (10, 11), (200, 13), (20, 15), (15, 17)]
data[(0, 60, 0)] = [(33, 2), (15, 4), (30, 7), (15, 8), (50, 9), (4.5, 10), (33, 17)]
data[(0, 60, 200)] = [(14, 3), (20, 4), (25, 8), (5.5, 10), (40, 15), (35, 16), (50, 18)]
data[(0, 60, 245)] = [(60, 1), (30, 4), (50, 9), (30, 11), (35, 15), (33, 17)]
data[(120, 20, 0)] = [(2, 3), (6, 5), (2, 6), (60, 13), (20, 16), (8, 18)]
data[(120, 20, 200)] = [(1 ,3), (2, 4), (5, 5), (2, 7), (.5, 10), (15, 16), (12, 18)]
data[(120, 20, 245)] = [(6, 2), (1, 3), (2, 5), (2, 6), (5, 9), (.5, 10), (10, 11), (15, 12), (8, 17)]
data[(120, 40, 0)] = [(20, 1), (10, 2), (20, 5), (12, 7), (10, 11), (120, 13), (15, 14), (10, 15), (20, 16), (20, 17), (24, 18)]
data[(120, 40, 200)] = [(15, 1), (20, 5), (12, 7), (20, 9), (15, 11), (10, 15), (22, 18)]
data[(120, 40, 245)] = [(30, 1), (5, 4), (20, 12), (80, 13)]
data[(120, 60, 0)] = [(50, 5), (18, 6), (50, 9), (2, 10)]
data[(120, 60, 200)] = [(28, 2), (22, 4), (18, 6), (20, 11)]
data[(120, 60, 245)] = [(20, 6), (25, 14), (20, 16), (28, 17)]
data[(220, 20, 0)] = [(3, 1), (3, 4), (1, 8), (.7, 10), (18, 12), (50, 13), (2, 15), (15, 17)]
data[(220, 20, 200)] = [(12, 16)]
data[(220, 20, 245)] = [(5, 1), (3, 2), (5, 6), (1, 8), (10, 14), (5, 15), (3, 18)]
data[(220, 40, 0)] = [(20, 1), (10, 2), (8, 8), (15, 11), (20, 12), (120, 13), (15, 16), ]
data[(220, 40, 200)] = [(20 ,12)]
data[(220, 40, 245)] = [(10, 6), (5, 10), (15, 14)]
data[(220, 60, 0)] = [(50, 1), (24, 2), (18, 3), (15, 8), (25, 12), (200, 13), (30, 14), (30, 16)]
data[(220, 60, 200)] = [(40, 1), (15, 3), (20, 7), (15, 8), (50, 9)]
data[(220, 60, 245)] = [(50, 5), (18, 6), (30, 7), (20, 8), (25, 14), (30, 16), (33, 17)]

# data gathered for each cue -- these will consist only of depth values -- we will need to convert to histogram to fit guassian

angle_1_vals = []
angle_2_vals = []
angle_3_vals = []
texture_1_vals = []
texture_2_vals = []
texture_3_vals = []
contrast_1_vals = []
contrast_2_vals = []
contrast_3_vals = []

for image in data:
	if image[0] == 0: # blur = 0 -- texture 1
		for entry in data[image]:
			texture_1_vals.append(entry[0]) # add on each depth response, ignoring the id for now
	elif image[0] == 120: # blur = 120 -- texture 2
		for entry in data[image]:
			texture_2_vals.append(entry[0])
	elif image[0] == 220: # blur = 220 -- texture 3
		for entry in data[image]:
			texture_3_vals.append(entry[0])
	if image[1] == 20: # angle = 20 -- angle 1
		for entry in data[image]:
			angle_1_vals.append(entry[0])
	elif image[1] == 40: # angle = 40 -- angle 2
		for entry in data[image]:
			angle_2_vals.append(entry[0])
	elif image[1] == 60: # angle = 60 -- angle 3
		for entry in data[image]:
			angle_3_vals.append(entry[0])
	if image[2] == 0: # color = 0 -- contrast 1
		for entry in data[image]:
			contrast_1_vals.append(entry[0])
	elif image[2] == 200: # color = 200 -- contrast 2
		for entry in data[image]:
			contrast_2_vals.append(entry[0])
	elif image[2] == 245: # color = 245 -- contrast 3
		for entry in data[image]:
			contrast_3_vals.append(entry[0])

# let's try getting rid of a1 outliers before we really get going
a1_cutoff = [i for i in angle_1_vals if 2<=i<=10]
a2_cutoff = [i for i in angle_2_vals if i<=40]
a3_cutoff = [i for i in angle_3_vals if 0<=i<=40]
t1_cutoff = [i for i in texture_1_vals if i<=40]
t2_cutoff = [i for i in texture_2_vals if i<=40]
t3_cutoff = [i for i in texture_3_vals if i<=40]
c1_cutoff = [i for i in contrast_1_vals if i<=40]
c2_cutoff = [i for i in contrast_2_vals if i<=40]
c3_cutoff = [i for i in contrast_3_vals if i<=40]

def gaussian(x, a, sig, mean):
	'''
	will use to fit a gaussian to each of the histogram distributions we end up with 
	'''
	return a/(sig*np.sqrt(2*math.pi))*np.exp(-1/2*((x-mean)/sig)**2)

# need to convert vals to histograms so that we have frequencies as a function of depth windows
# histogram returns (values of histogram, bin_edges)
num_bins = 8
a1_hist, a1_bin_bounds = np.histogram(a1_cutoff, bins=num_bins) # fit to the peak [1:6] with 8 bins -- people have bias for no change in depth
a1_bins = a1_bin_bounds + (a1_bin_bounds[1]-a1_bin_bounds[0])/2
a1_params, a1_pcov = curve_fit(gaussian, a1_bins[1:6], a1_hist[1:6]) # this gives us good coefficients
num_bins = 6
a2_hist, a2_bins = np.histogram(a2_cutoff, bins=num_bins) # 6 bins works 
a2_bins = a2_bins + (a2_bins[1]-a2_bins[0])/2
a2_params, a2_pcov = curve_fit(gaussian, a2_bins[0:num_bins], a2_hist) # this gives us good coefficients
num_bins = 8
a3_hist, a3_bins = np.histogram(a3_cutoff, bins=num_bins) # use 8 bins 
a3_bins = a3_bins + (a3_bins[1]-a3_bins[0])/2
a3_params, a3_pcov = curve_fit(gaussian, a3_bins[0:num_bins], a3_hist) # this gives us good coefficients with 8 bins

################# we are going to just add in high variance distributions for texture and contrast ##########################
# t1_hist, t1_bins = np.histogram(t1_cutoff, bins=num_bins) # might need to cutoff more -- works okay with 8 bins for now
# t2_hist, t2_bins = np.histogram(t2_cutoff, bins=num_bins) # 8 bins seems to work decently
# t3_hist, t3_bins = np.histogram(t3_cutoff, bins=num_bins) # 8 bins seems to work pretty well
# c1_hist, c1_bins = np.histogram(c1_cutoff, bins=num_bins) # 8 bins seems to work okay
# c2_hist, c2_bins = np.histogram(c2_cutoff, bins=num_bins) # 8 bins works well
# c3_hist, c3_bins = np.histogram(c3_cutoff, bins=num_bins) # 7 bins works well
#################################################################################################################

################################## figure to demonstrate angle gaussian ##########################################
# angle_hist = plt.hist(a1_cutoff, normed=True, bins=a1_bin_bounds[1:8])
# plt.title('P(d|a) for 20 degree perspective angle', fontsize='x-large')
# plt.xlabel('Depth', fontsize='x-large')
# plt.ylabel('P(d|a)', fontsize='x-large')
# x = np.linspace(2, 10, 100)
# y = gaussian(x, 1, a1_params[1], a1_params[2])
# plt.plot(x, y, 'r-')
# plt.show()
##################################################################################################################


# since we can't seem to get great fits for texture and contrast, we will approximate them with a high variance Gaussian
# fit centered at the mean of the distribution
t1_mean, t1_std = np.mean(t1_cutoff), np.std(t1_cutoff)
t2_mean, t2_std = np.mean(t2_cutoff), np.std(t2_cutoff)
t3_mean, t3_std = np.mean(t3_cutoff), np.std(t3_cutoff)
c1_mean, c1_std = np.mean(c1_cutoff), np.std(c1_cutoff)
c2_mean, c2_std = np.mean(c2_cutoff), np.std(c2_cutoff)
c3_mean, c3_std = np.mean(c3_cutoff), np.std(c3_cutoff)
t1_params = [None, t1_std, t1_mean]
t2_params = [None, t2_std, t2_mean]
t3_params = [None, t3_std, t3_mean]
c1_params = [None, c1_std, c1_mean]
c2_params = [None, c2_std, c2_mean]
c3_params = [None, c3_std, c3_mean]

################################ figure to demonstrate lack of coherence in texture ################################
# texture_hist = plt.hist(t2_cutoff, normed=True, bins=7)
# plt.title('P(d|t) for high blurring', fontsize='x-large')
# plt.xlabel('Depth', fontsize='x-large')
# plt.ylabel('P(d|t)', fontsize='x-large')
# x = np.linspace(0, 50, 200)
# y = gaussian(x, 1, t2_std, t2_mean)
# plt.plot(x, y, 'r-')
# plt.show()

# ############################## figure to demonstrate lack of coherence in color ####################################
# color_hist = plt.hist(c3_cutoff, normed=True, bins=7)
# plt.title('P(d|c) for low contrast', fontsize='x-large')
# plt.xlabel('Depth', fontsize='x-large')
# plt.ylabel('P(d|c)', fontsize='x-large')
# x = np.linspace(0, 50, 200)
# y = gaussian(x, 1, c3_std, c3_mean)
# plt.plot(x, y, 'r-')
# plt.show()
##############################################################################################################################
# now for Kalman filter

params_dict = {'a1': a1_params, 'a2': a2_params, 'a3': a3_params, 't1': t1_params, 't2': t2_params, 't3': t3_params, 'c1': c1_params, 'c2': c2_params, 'c3': c3_params}

def optimal_estimate(angle, texture, contrast):
    '''
    takes in each cue for the image and returns the optimal depth percept
    cues must be in string form as they are in the params_dict
    '''
    sig_a = params_dict[angle][1]
    d_a = params_dict[angle][2]
    sig_t = params_dict[texture][1]
    d_t = params_dict[texture][2]
    sig_c = params_dict[contrast][1]
    d_c = params_dict[contrast][2]
    
    # first need to calculate the weights
    w_a = (1/sig_a**2)/(1/sig_a**2 + 1/sig_t**2 + 1/sig_c**2)
    w_t = (1/sig_t**2)/(1/sig_a**2 + 1/sig_t**2 + 1/sig_c**2)
    w_c = (1/sig_c**2)/(1/sig_a**2 + 1/sig_t**2 + 1/sig_c**2)
    
    # now to calculate the optimal depth
    d_star = w_a*d_a + w_t*d_t + w_c*d_c
    return d_star

############################## plot a1 gaussian, t1 gaussian, c1 gaussian, as well as optimal depth and average depth ##############################
gaussians = plt.figure()
x1 = np.linspace(0, 20, 200)
y1 = gaussian(x1, 1, a2_params[1], a2_params[2])
plt.plot(x1, y1, 'r-', label='P(d|a)')

x2 = np.linspace(0, 30, 300)
y2 = gaussian(x2, 1, t1_params[1], t1_params[2])
plt.plot(x2, y2, 'b-', label='P(d|t)')

x3 = np.linspace(0, 30, 300)
y3 = gaussian(x3, 1, c1_params[1], c1_params[2])
plt.plot(x3, y3, 'g-', label='P(d|c)')

optimal = optimal_estimate('a2', 't1', 'c1')
average = np.mean([a2_params[2], t1_params[2], c1_params[2]])

plt.axvline(x=optimal, color='k', linestyle='-', label='Optimal estimate')
plt.axvline(x=average, color='k', linestyle='--', label='Cue averaging')
plt.legend(fontsize='x-large')
plt.xlabel('Depth', fontsize='x-large')
plt.ylabel('Probability', fontsize='x-large')
plt.title('Depth probability from three different cues', fontsize='x-large')
plt.show()

############################# plot depth predictions as angle changes while holding other cues constant ###############################
# angle_estimates = []
# for angle in ['a1', 'a2', 'a3']:
# 	angle_estimates.append(optimal_estimate(angle, 't1', 'c1'))

# # need to calculate mean depth observation for each angle value with other cues at 1
# angle_observations = []
# angle_dict = {} # keys are angles, object is array of tuples of form (observation, i)
# for entry in data:
# 	if entry == (0, 20, 0): # 20 degree angle
# 		angle_dict[20] = data[entry]
# 	elif entry == (0, 40, 0): # 40 degree angle
# 		angle_dict[40] = data[entry]
# 	elif entry == (0, 60, 0): # 60 degree angle
# 		angle_dict[60] = data[entry]

# aggregate = []
# for pair in angle_dict[20]:
# 	if pair[0] < 60: # remove outliers
# 		aggregate.append(pair[0])
# angle_observations.append(np.mean(aggregate))

# aggregate = []
# for pair in angle_dict[40]:
# 	aggregate.append(pair[0])
# angle_observations.append(np.mean(aggregate))

# aggregate = []
# for pair in angle_dict[60]:
# 	if pair[0] < 50: # remove outliers
# 		aggregate.append(pair[0])
# angle_observations.append(np.mean(aggregate))


# x = [20, 40, 60]
# fig1 = plt.figure()
# averages = [np.mean([a1_params[2], t1_params[2], c1_params[2]]), np.mean([a2_params[2], t1_params[2], c1_params[2]]), np.mean([a3_params[2], t1_params[2], c1_params[2]])]
# plt.plot(x, angle_estimates, 'bx', label='Optimal depth estimates', markersize=12)
# plt.plot(x, angle_observations, 'ro', label='Depth observations', markersize=12)
# plt.plot(x, averages, 'k+', label='Averaged cue estimates', markersize=12)
# plt.title('Depth estimates and observations as a function of angle', fontsize='x-large')
# plt.legend(fontsize='x-large')
# plt.xlabel('Angle of Perspective', fontsize='x-large')
# plt.ylabel('Depth', fontsize='x-large')
# plt.show()

#############################################################################################################################
####################### plot depth predictions as texture changes while holding other cues constant #########################
# texture_estimates = []
# for texture in ['t1', 't2', 't3']:
# 	texture_estimates.append(optimal_estimate('a3', texture, 'c2'))

# # need to calculate mean depth observation for each texture value with other cues at a2 and c2
# texture_observations = []
# texture_dict = {} # keys are textures, object is array of tuples of form (observation, i)
# for entry in data:
# 	if entry == (0, 60, 200): # 0 blurring
# 		texture_dict[0] = data[entry]
# 	elif entry == (120, 60, 200): # 120 blurring
# 		texture_dict[120] = data[entry]
# 	elif entry == (220, 60, 200): # 220 blurring
# 		texture_dict[220] = data[entry]

# aggregate = []
# for pair in texture_dict[0]:
# 	if pair[0] < 35:
# 		aggregate.append(pair[0])
# texture_observations.append(np.mean(aggregate))

# aggregate = []
# for pair in texture_dict[120]:
# 	if pair[0] < 35:
# 		aggregate.append(pair[0])
# texture_observations.append(np.mean(aggregate))

# aggregate = []
# for pair in texture_dict[220]:
# 	if pair[0] < 35:
# 		aggregate.append(pair[0])
# texture_observations.append(np.mean(aggregate))


# x = [0, 120, 220]
# fig1 = plt.figure()
# averages = [np.mean([a3_params[2], t1_params[2], c2_params[2]]), np.mean([a3_params[2], t2_params[2], c2_params[2]]), np.mean([a3_params[2], t3_params[2], c2_params[2]])]
# plt.plot(x, texture_estimates, 'bx', label='Optimal depth estimates', markersize=12)
# plt.plot(x, texture_observations, 'ro', label='Depth observations', markersize=12)
# plt.plot(x, averages, 'k+', label='Averaged cue estimates', markersize=12)
# plt.title('Depth estimates and observations as a function of texture', fontsize='x-large')
# plt.legend(fontsize='x-large')
# plt.xlabel('Amount of blur', fontsize='x-large')
# plt.ylabel('Depth', fontsize='x-large')
# plt.show()

#############################################################################################################################
####################### plot depth predictions as contrast changes while holding other cues constant #########################
# contrast_estimates = []
# for contrast in ['c1', 'c2', 'c3']:
# 	contrast_estimates.append(optimal_estimate('a3', 't2', contrast))

# # need to calculate mean depth observation for each texture value with other cues at a2 and c2
# contrast_observations = []
# contrast_dict = {} # keys are textures, object is array of tuples of form (observation, i)
# for entry in data:
# 	if entry == (120, 60, 0): # full contrast
# 		contrast_dict[0] = data[entry]
# 	elif entry == (120, 60, 200): # medium contrast
# 		contrast_dict[200] = data[entry]
# 	elif entry == (120, 60, 245): # low contrast
# 		contrast_dict[245] = data[entry]

# aggregate = []
# for pair in contrast_dict[0]:
# 	if 2 < pair[0] < 35:
# 		aggregate.append(pair[0])
# contrast_observations.append(np.mean(aggregate))

# aggregate = []
# for pair in contrast_dict[200]:
# 	if pair[0] < 35:
# 		aggregate.append(pair[0])
# contrast_observations.append(np.mean(aggregate))

# aggregate = []
# for pair in contrast_dict[245]:
# 	if pair[0] < 35:
# 		aggregate.append(pair[0])
# contrast_observations.append(np.mean(aggregate))


# x = [0, 200, 245]
# fig1 = plt.figure()
# averages = [np.mean([a3_params[2], t2_params[2], c1_params[2]]), np.mean([a3_params[2], t2_params[2], c2_params[2]]), np.mean([a3_params[2], t2_params[2], c3_params[2]])]
# plt.plot(x, contrast_estimates, 'bx', label='Optimal depth estimates', markersize=12)
# plt.plot(x, contrast_observations, 'ro', label='Depth observations', markersize=12)
# plt.plot(x, averages, 'k+', label='Averaged cue estimates', markersize=12)
# plt.title('Depth estimates and observations as a function of contrast', fontsize='x-large')
# plt.legend(fontsize='x-large')
# plt.xlabel('Decrease in contrast', fontsize='x-large')
# plt.ylabel('Depth', fontsize='x-large')
# plt.show()


