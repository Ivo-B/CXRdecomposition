'''
We assume for simplicity a parallel projection model and compute the average attenuation
coefficient along the y axis ranging from [1,N] (where N is the pixel length of the
posterior anterior view). Denoting the CT volume by G(x,y,z), the 2D average
attenuation map can be computed using equation 2:
'''
# load ct data
import numpy as np

input_shape = ct_data.shape

drr_out = np.array((input_shape[0], input_shape[2]), dtype=np.float32)
for x in input_shape[0]:
	for z in input_shape[2]:
		u_av = 0.0
		for y in input_shape[1]:
			u_av += 0.2 * (ct_data[x, y, z] + 1000) / (input_shape[1] * 1000)
		drr_out = np.exp(0.02 + u_av)
