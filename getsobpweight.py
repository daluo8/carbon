import numpy as np
import pickle
import torch
import torch.optim
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


# def dataloader(path):
#     with open(path, 'rb')as f:
#         data = pickle.load(f)
#     return data

def getsobpweight(data, R, resolutionz, sigma):
    Rmin, Rmax = R[0], R[-1]
    data_ranges = np.squeeze(np.array(data['range']))
    dose_arr = []
    for i in range(0, len(R)):
        idx = np.squeeze(np.where(data_ranges == R[i] / 10))
        depthdose = data['dose'][idx]
        dose_norm = depthdose / np.max(depthdose)
        dose1D = np.zeros(dose_norm.shape[1])
        for j in range(-sigma, sigma):
            curve_weight = 1 / (np.sqrt(2 * math.pi) * sigma) * np.exp(i ** 2 / (2 * sigma ** 2))
            if j < 0:
                dose_temp = dose_norm[0][-j:dose_norm.shape[1]]
                for k in range(0, -j):
                    dose_temp = np.append(dose_temp, 2 * dose_temp[-1] - dose_temp[-2])
            elif j > 0:
                dose_temp = dose_norm[0][0:dose_norm.shape[1] - j]
                for k in range(0, j):
                    dose_temp = np.append(2 * dose_temp[0] - dose_temp[1], dose_temp)
            else:
                dose_temp = dose_norm
            dose1D = dose1D + curve_weight * dose_temp
        dose1D = np.squeeze(dose1D) / np.max(dose1D)
        dose_arr.append(dose1D)
    dose_arr = np.squeeze(np.array(dose_arr))
    w0 = np.ones((1, len(R)))

    def sobp_weight(w):
        fn = np.sum(np.matmul(w, dose_arr[:, 0:int(R[0]) + resolutionz - 1])) / (R[0] + resolutionz - 1)
        ft = np.sum(np.matmul(w, dose_arr[:, int(R[0]) + resolutionz:int(R[-1]) - resolutionz])) / (
                R[-1] - R[0] - 2 * resolutionz)
        r = fn / ft
        miu = ft
        s = pow(np.matmul(w, dose_arr[:, int(R[0]) + resolutionz:int(R[-1]) - resolutionz]) - miu, 2).sum() / ft
        f = r * s
        # print(f)
        return f

    weight_bound = (w0.shape[1] - 1) * ((0.05, 1),)
    weight_bound = weight_bound + ((0.99, 1),)
    result = minimize(fun=sobp_weight, x0=w0, bounds=weight_bound)
    weights=result.x/np.max(result.x)
    return weights #sobp weights
