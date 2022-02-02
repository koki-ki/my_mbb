# coding: utf-8
import numpy as np
from scipy import stats

def estimate_scipy_gaussian_kernel_density(hms, h=None, eta=0.5, weight_iter_M=20):
    hms = hms.flatten()
    #"""
    weights = np.ones_like(hms)
    weight_iter = 0
    h = -1
    while weight_iter < weight_iter_M:
        parzen = stats.gaussian_kde(dataset=hms, weights=weights)
        p = parzen(hms)
        log_p = np.log(p)
        log_g = np.sum(log_p) / len(hms)
        weights = np.exp(-eta * (log_p - log_g) )
        h = parzen.factor
        weight_iter += 1
    h_w = weights * h
    return np.reshape(np.array(h_w), (-1, 1))

def make_one_class_parzen_width_init_with_IQR(hms):
    hms = np.array(hms).flatten()
    sorted_d = sorted(hms)
    mean_d = np.mean(hms)
    std_d = np.std(hms)
    lower_idx = len(hms) // 4
    upper_idx = len(hms) - lower_idx
    IQR = sorted_d[upper_idx] - sorted_d[lower_idx]
    h = np.power((4.0 / 3.0), (1.0 / 5.0)) * np.minimum(std_d, (IQR / 1.34)) * np.power(float(len(hms)), (-1.0 / 5.0))
    return h

def estimate_one_class_parzen_width(hms, h=None, max_loop=100):
    hms = np.array(hms).flatten()
    N_d = len(hms)
    differences = np.reshape(hms, (N_d, 1)) - hms  # N * N, 0 on diagonal line
    ma_differences = np.ma.masked_array(differences, mask=np.identity(N_d))

    distances_NN = np.amin(np.abs(ma_differences), axis=1)
    width_h = h
    if width_h is None:
        width_h = np.sqrt(np.sum(np.square(distances_NN)) / N_d)
        #width_h = make_one_class_parzen_width_init_with_IQR(hms)
    weight_w = np.ones(N_d, dtype=float)

    est_weight_iter = 0
    eta = 0.5
    while est_weight_iter < 10:
        h_w = weight_w * width_h
        log_estims = np.log(np.sum(np.exp(-0.5 * np.square(-differences / h_w)) / h_w, axis=1) / N_d)
        log_geom_mean = np.sum(log_estims) / N_d
        weight_w = np.exp((log_estims - log_geom_mean) * eta)

        est_width_iter = 0
        while est_width_iter < max_loop:
            h_w = weight_w * width_h
            #F.append(-N_d * np.log(width_h) + np.sum(np.(np.sum(np.exp(-0.5 * np.square(-ma_differences / h_w)) / weight_w, axis=1))))

            # Estep
            inner_exp = -0.5 * np.square(ma_differences / h_w)
            inner_exp_max = np.amax(inner_exp, axis=1)[:, np.newaxis]
            log_num_q = inner_exp - inner_exp_max - np.log(h_w)
            log_denom_q = np.log(np.sum(np.exp(log_num_q), axis=1))[:, np.newaxis]
            q = np.exp(log_num_q - log_denom_q)

            # Mstep
            width_h = np.sqrt(np.sum(q * np.square(ma_differences)) / N_d)

            est_width_iter += 1
        est_weight_iter += 1

    h_w = weight_w * width_h
    return np.reshape(np.array(h_w), (-1, 1))

def estimate_loss_smoothness(h):
    return (4.0 / (np.sqrt(2.0 * np.pi) * h))

