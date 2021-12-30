import numpy as np

src_angle_incidence = 5 # degrees

src_hgt_Si = 3.5
src_hgt_polymer = 4
src_hgt_Cu = 2.8

n_Si = 3.43
n_polymer = 1.5
n_Cu = 2.718

heights = [0, src_hgt_polymer, src_hgt_Si, src_hgt_Cu]
indices = [1, n_polymer, n_Si, n_Cu]

def gaussRefrOffset(src_angle_incidence, heights, indices):
# structured such that n0 is the input interface and n_end is the last above the device

    src_angle_incidence = np.radians(src_angle_incidence)
    snellConst = indices[-1]*np.sin(src_angle_incidence)
    input_theta = np.degrees(np.arcsin(snellConst/indices[0]))

    r_offset = 0
    for cnt, h_i in enumerate(heights):
        s_i = snellConst/indices[cnt]
        r_offset = r_offset + h_i*s_i/np.sqrt(1-s_i**2)

    return [input_theta, r_offset]

def adjustGaussRefract(src_angle_incidence, src_hgt_Si, src_hgt_polymer, n_Si, n_polymer):
#todo(ianfoo): edit this so that it takes n entries and not just two
	r_offset = src_hgt_Si * np.tan(src_angle_incidence*np.pi/180)
	sin_th_prime = n_polymer/n_Si * np.sin(src_angle_incidence*np.pi/180)
	r_offset = r_offset + src_hgt_polymer*sin_th_prime/np.sqrt(1-sin_th_prime**2)
	input_theta = np.arcsin(sin_th_prime) * 180/np.pi

	return [input_theta, r_offset]

x = adjustGaussRefract(src_angle_incidence, src_hgt_Si, src_hgt_polymer, n_Si, n_polymer)
y = gaussRefrOffset(src_angle_incidence, heights, indices)