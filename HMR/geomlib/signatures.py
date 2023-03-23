"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import numpy as np

# the heat kernel signature
# implemented as described in Sun et al. 2009
def compute_HKS(eigen_vecs, eigen_vals, num_t, t_min=0.1, t_max=1000, scale=1000):
    eigen_vals = eigen_vals.flatten()
    assert eigen_vals[1] > 0
    assert np.min(eigen_vals) > -1E-6
    assert np.array_equal(eigen_vals, sorted(eigen_vals))

    t_list = np.geomspace(t_min, t_max, num_t)
    phase = np.exp(-np.outer(t_list, eigen_vals[1:]))
    wphi = phase[:, None, :] * eigen_vecs[None, :, 1:] 
    HKS = np.einsum('tnk,nk->nt', wphi, eigen_vecs[:, 1:]) * scale
    heat_trace = np.sum(phase, axis=1)
    HKS /= heat_trace

    return HKS


# the wave kernel signature
# implemented as described in Aubry et al. 2011
def compute_WKS(eigen_vecs, eigen_vals, num_E, E_min=np.log(1E-4), E_max=np.log(0.4), scale=1000):
    eigen_vals = eigen_vals.flatten()
    assert eigen_vals[1] > 0
    assert np.min(eigen_vals) > -1E-6
    assert np.array_equal(eigen_vals, sorted(eigen_vals))

    sigma = (E_max - E_min) / num_E
    E_min += 2 * sigma
    E_max -= 2 * sigma
    E_list = np.linspace(E_min, E_max, num_E)

    indices = np.where(eigen_vals > 1e-5)[0]
    eigen_vals = eigen_vals[indices]
    eigen_vecs = eigen_vecs[:, indices]

    coeffs = np.exp(-np.square(E_list[:, None] - np.log(eigen_vals)[None, :]) / (2 * sigma**2))
    wphi = eigen_vecs[None, :, :] * coeffs[:, None, :]
    WKS = np.einsum('enk,nk->ne', wphi, eigen_vecs)
    ce = 1 / np.sum(coeffs, axis=1)
    
    return ce[None, :] * WKS * scale


