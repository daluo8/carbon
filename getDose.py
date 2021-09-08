# import pandas as pd
import numpy as np
import pickle
import cmath
import time
import math
import matplotlib.pyplot as plt

import numba as nb


def dataloader(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data


def calculate3Ddose(data, centrax, centray, R, resolution):
    Rcsda_arr = np.squeeze(np.array(data['range']))
    energyidx = np.squeeze(np.where(Rcsda_arr == R))
    attributes = ['range', 'energy', 'dose', 'weight2', 'weight3', 'sigma', 'sf1', 'sf2', 'sf3']
    metadata = {attr: [] for attr in attributes}
    for attr in attributes:
        metadata[attr] = np.squeeze(data[attr][energyidx])
    '''
    depthdose:一维深度剂量
    weight2:第二项高斯权重
    weight3:第三项高斯权重
    sigma:单高斯sigma
    sf1:第一项高斯修正因子
    sf2:第二项高斯修正因子
    sf3:第三项高斯修正因子
    e.g. sigma1=sigma*sf1
    '''
    depthdose, weight2, weight3 = metadata['dose'], metadata['weight2'], metadata['weight3']

    sf1, sf2, sf3, sigma = metadata['sf1'], metadata['sf2'], metadata['sf3'], metadata['sigma']

    # 横向的计算范围为3.5倍MCSsigma
    # rmax = 3.5 * metadata['MCSsigma']
    # distal 为束流深度方向计算的最大值，超过这个深度认为其剂量已经可以忽略不计
    # rangestruggle = 0.012 * pow(R, 0.935) * 10
    # distal = round(R * 10 + 5 * rangestruggle)

    @nb.jit(nopython=True)
    def cal(depthdose, weight2, weight3, sigma, sf1, sf2, sf3):
        # dose = np.zeros((resolution[0], resolution[1], resolution[2]))
        dose = np.zeros((dimx, dimy, dimz))
        for i in range(0, dimz):
            for j in range(0, dimx):
                for k in range(0, dimy):
                    r2 = (j - centrax) ** 2 + (k - centray) ** 2
                    # if r2 <= rmax[i] ** 2:
                    # print(metadata['dose'])
                    dose[j, k, i] = ((1 - np.exp(weight2[i]) - np.exp(weight3[i])) * np.exp( \
                        -r2 / (2 * sf1[i] * sigma[i] ** 2) / (2 * math.pi * sf1[i] * sigma[i] ** 2)) \
                                                    + np.exp(weight2[i]) * np.exp(
                                -r2 / (2 * (3 * sf2[i] * sigma[i]) ** 2) / (2 * math.pi * (3 * sf2[i] * sigma[i]) ** 2)) \
                                                    + np.exp(weight3[i]) * np.exp(
                                -r2 / (2 * (5 * sf3[i] * sigma[i]) ** 2) / (
                                        2 * math.pi * (5 * sf2[i] * sigma[i]) ** 2)))
            dose[:, :, i] = depthdose[i] * dose[:, :, i] / np.sum(dose[:, :, i])
        return dose

    return cal(depthdose, weight2, weight3, sigma, sf1, sf2, sf3), depthdose


centrax = 80 + .5  # 束流中心的x,y坐标
centray = 80 + .5
inisigma = 3  # 束流初始的横向sigma[mm]
# angularspread = 1  # mrad
# SSD = 1000  # 源到皮肤（模体表面）的距离 [mm]
R = 30  # 碳离子在水中的射程
path = './carbontrigauss.pkl'
data = dataloader(path)
start = time.time()
dimx, dimy, dimz = 160, 160, 400
ctresolution = [dimx, dimy, dimz]
dose, depthdose = calculate3Ddose(data, centrax, centray, R, ctresolution)
depthdose = depthdose / max(depthdose)
print("time usage: %fsec" % (time.time() - start))
dosezx = np.sum(dose, axis=0)
dosezx_norm = dosezx / np.max(dosezx)
plt.figure()
im = plt.contourf(dosezx_norm, cmap=plt.cm.jet)
plt.contour(dosezx)

plt.colorbar(im)

plt.figure()
dose1D = np.sum(dosezx, axis=0)
dose1D_norm = dose1D / max(dose1D)
xarr = np.linspace(1, dimz, dimz)
plt.plot(xarr, dose1D_norm, label='IDD')
# plt.plot(xarr, depthdose, label='MC')
plt.xlim(0, dimz)
plt.ylim(0, 1)
plt.legend()
plt.show()
