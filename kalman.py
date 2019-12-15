import numpy as np 
import math
from scipy.optimize import curve_fit

# data gathered for each cue -- these will consist only of depth values -- we will need to convert to histogram to fit guassian
angle_1_vals = np.array()
angle_2_vals = np.array()
angle_3_vals = np.array()
texture_1_vals = np.array()
texture_2_vals = np.array()
texture_3_vals = np.array()
contrast_1_vals = np.array()
contrast_2_vals = np.array()
contrast_3_vals = np.array()

# need to convert vals to histograms so that we have frequencies as a function of depth windows
# histogram returns (values of histogram, bin_edges)
a1_hist, a1_bins = np.histogram(angle_1_vals)
a2_hist, a2_bins = np.histogram(angle_2_vals)
a3_hist, a3_bins = np.histogram(angle_3_vals)
t1_hist, t1_bins = np.histogram(texture_1_vals)
t2_hist, t2_bins = np.histogram(texture_2_vals)
t3_hist, t3_bins = np.histogram(texture_3_vals)
c1_hist, c1_bins = np.histogram(contrast_1_vals)
c2_hist, c2_bins = np.histogram(contrast_2_vals)
c3_hist, c3_bins = np.histogram(contrast_3_vals)

a1_bins = a1_bins + (a1_bins[1]-a1_bins[0])/2
a2_bins = a2_bins + (a2_bins[1]-a2_bins[0])/2
a3_bins = a3_bins + (a3_bins[1]-a3_bins[0])/2
t1_bins = t1_bins + (t1_bins[1]-t1_bins[0])/2
t2_bins = t2_bins + (t2_bins[1]-t2_bins[0])/2
t3_bins = t3_bins + (t3_bins[1]-t3_bins[0])/2
c1_bins = c1_bins + (c1_bins[1]-c1_bins[0])/2
c2_bins = c2_bins + (c2_bins[1]-c2_bins[0])/2
c3_bins = c3_bins + (c3_bins[1]-c3_bins[0])/2


def gaussian(x, a, sig, mean):
	'''
	will use to fit a gaussian to each of the histogram distributions we end up with 
	'''
	return a/(sig*np.sqrt(2*math.pi))*np.exp(-1/2*((x-mean)/sig)**2)


# gaussian distributions for all P(d|a), P(d|t), and P(d|c) -- we will fit these from the data we gather for each cue independent of the others
a1_params, a1_pcov = curve_fit(gaussian, a1_bins[0:10], a1_hist)
a2_params, a2_pcov = curve_fit(gaussian, a2_bins[0:10], a2_hist)
a3_params, a3_pcov = curve_fit(gaussian, a3_bins[0:10], a3_hist)
t1_params, t1_pcov = curve_fit(gaussian, t1_bins[0:10], t1_hist)
t2_params, t2_pcov = curve_fit(gaussian, t2_bins[0:10], t2_hist)
t3_params, t3_pcov = curve_fit(gaussian, t3_bins[0:10], t3_hist)
c1_params, c1_pcov = curve_fit(gaussian, c1_bins[0:10], c1_hist)
c2_params, c2_pcov = curve_fit(gaussian, c2_bins[0:10], c2_hist)
c3_params, c3_pcov = curve_fit(gaussian, c3_bins[0:10], c3_hist)

##############################################################################################################################
# now for Kalman filter



