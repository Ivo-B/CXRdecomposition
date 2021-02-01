import numpy as np
from numba import jit

@jit(nopython=True, parallel=True)
def generate_drr_from_ct(ct_scan, direction='top'):
    input_shape = ct_scan.shape
    if direction == 'lateral':
        ct_scan = np.transpose(ct_scan, axes=(0, 2, 1))
        input_shape = ct_scan.shape
    elif direction == "frontal":
        ct_scan = np.transpose(ct_scan, axes=(1, 0, 2))
        input_shape = ct_scan.shape

    drr_out = np.zeros((input_shape[0], input_shape[2]), dtype=np.float32)
    for x in range(input_shape[0]):
        for z in range(input_shape[2]):
            u_av = 0.0
            for y in range(input_shape[1]):
                u_av += 0.2 * (ct_scan[x, y, z] + 1000) / (input_shape[1] * 1000)
            drr_out[x, z] = np.exp(0.02 + u_av)
    return drr_out



def do_full_prprocessing(ct_data, input_views):
    drr_front = generate_drr_from_ct(ct_data, direction='frontal')
    drr_lat = None
    drr_top = None
    if input_views > 1:
        drr_lat = generate_drr_from_ct(ct_data, direction='lateral')
        #rotate lat to alinge with frontal projection
        drr_lat = np.transpose(drr_lat, (1, 0))
    if input_views > 2:
        drr_top = generate_drr_from_ct(ct_data, direction='top')

    return [drr_front, drr_lat, drr_top]


