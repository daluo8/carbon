import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from prepare_phantom import *
from getDose import *
from getsobpweight import *


class weightIni(torch.nn.Module):
    def __init__(self):
        # 初始化loss参数
        super(weightIni, self).__init__()
        self.xDim = 100
        self.yDim = 100
        self.zDim = 300
        self.xPTV = self.xDim / 2
        self.yPTV = self.yDim / 2
        self.zPTV = 100

        self.center = np.array([self.xPTV, self.yPTV, self.zPTV])
        self.radiusPTV = 5
        self.PTV_margin = 3
        self.radiusPTV = self.radiusPTV + self.PTV_margin

        self.resolutionx = 3
        self.resolutiony = 5
        self.resolutionz = 5
        self.resolution = [self.resolutionx, self.resolutiony, self.resolutionz]

        self.inisigma = 3  # 束流初始的横向sigma[mm]
        self.path = './carbontrigauss.pkl'

        self.doseobj = 100
        self.dosePTVmin = 0.95
        self.doseOARmax = 0.5

    def _weight_init(self):
        # 读入患者CT数据（demo为自定义模体）
        self.mycst = cst()
        PTVix = []
        OARix = []

        self.PTV = self.mycst.tissue_init(1, 'TARGET', PTVix)
        self.OAR = self.mycst.tissue_init(2, 'OAR', OARix)

        self.PTV.PTVix = get_PTVix(self.center, self.xDim, self.yDim, self.zDim, self.radiusPTV)
        self.OAR.OARix = get_OARix(self.center, self.xDim, self.yDim, self.zDim)

        # 读取3高斯数据
        self.data = dataloader(self.path)
        self.PTV_grid = np.transpose(np.array(self.PTV.PTVix))

        zmin, zmax = energy_selector(self.data, self.PTV_grid)
        if np.mod(zmin, self.resolutionz) < self.PTV_margin:
            Rmin = zmin - np.mod(zmin, self.resolutionz) - self.resolutionz
        else:
            Rmin = zmin - np.mod(zmin, self.resolutionz)
        if self.resolutionz - np.mod(zmax, self.resolutionz) < self.PTV_margin:
            Rmax = zmax + self.resolutionz - np.mod(zmax, self.resolutionz) + self.resolutionz
        else:
            Rmax = zmax + np.mod(zmax, self.resolutionz)

        PTV_temp = []
        scan_depths = np.linspace(Rmin, Rmax, (Rmax - Rmin) / self.resolutionz + 1)
        for scan_depth in scan_depths:
            # if scan_depth == Rmin:
            #     prox_idxs = np.where(self.PTV_grid[:, 2] == zmin)
            #     for prox_idx in prox_idxs:
            #         prox_point = [self.PTV_grid[prox_idx, 0][0], self.PTV_grid[prox_idx, 1][0], Rmin]
            #         PTV_temp.append(prox_point)
            #     if np.mod(zmin, self.resolutionz) < self.PTV_margin:
            #         for prox_idx in prox_idxs:
            #             prox_point = [self.PTV_grid[prox_idx, 0][0], self.PTV_grid[prox_idx, 1][0],
            #                           Rmin + self.resolutionz]
            #             PTV_temp.append(prox_point)
            if scan_depth < zmin:
                idxs = np.where(self.PTV_grid[:, 2] == zmin)
                for idx in idxs:
                    prox_point = [self.PTV_grid[idx, 0][0], self.PTV_grid[idx, 1][0], scan_depth]
                    PTV_temp.append(prox_point)
            elif scan_depth > zmax:
                idxs = np.where(self.PTV_grid[:, 2] == zmax)
                for idx in idxs:
                    dis_point = [self.PTV_grid[idx, 0][0], self.PTV_grid[idx, 1][0], scan_depth]
                    PTV_temp.append(dis_point)
                # dis_idxs = np.where(self.PTV_grid[:, 2] == zmax)
                # for dis_idx in dis_idxs:
                #     dis_point = [self.PTV_grid[dis_idx, 0][0], self.PTV_grid[dis_idx, 1][0], Rmax]
                #     PTV_temp.append(dis_point)
                #     if self.resolutionz - np.mod(zmax, self.resolutionz) < self.PTV_margin:
                #         for dis_idx in dis_idxs:
                #             dis_point = [self.PTV_grid[dis_idx, 0][0], self.PTV_grid[dis_idx, 1][0],
                #                          Rmax - self.resolutionz]
                #             PTV_temp.append(dis_point)
            else:
                slice_idxs = np.where(self.PTV_grid[:, 2] == scan_depth)
                slice_points = self.PTV_grid[slice_idxs, :][0]
                # print(len(slice_points))
                if len(slice_points) == 0:
                    continue

                Ymin = min(slice_points[:, 1])
                Ymax = max(slice_points[:, 1])
                num_of_Y = np.ceil((Ymax + 2 * self.PTV_margin - Ymin) / self.resolutiony) + 1
                for coord_y in np.linspace(Ymin - self.PTV_margin, Ymax + self.PTV_margin, num_of_Y):
                    if coord_y < Ymin:
                        idxs = np.where(slice_points[:, 1] == Ymin)
                        for idx in idxs:
                            temp_point = [slice_points[idx, 0][0], coord_y, scan_depth]
                            PTV_temp.append(temp_point)
                    elif coord_y > Ymax:
                        idxs = np.where(slice_points[:, 1] == Ymax)
                        for idx in idxs:
                            temp_point = [slice_points[idx, 0][0], coord_y, scan_depth]
                            PTV_temp.append(temp_point)
                    else:
                        lat_point_idx = np.where(slice_points[:, 1] == round(coord_y))
                        lat_points = slice_points[lat_point_idx, :][0]
                        Xmin = min(lat_points[:, 0]) - self.PTV_margin
                        Xmax = max(lat_points[:, 0]) + self.PTV_margin
                        num_of_X = np.ceil((Xmax - Xmin) / self.resolutionx) + 1
                        for coord_x in np.linspace(Xmin, Xmax, num_of_X):
                            PTV_temp.append([coord_x, coord_y, scan_depth])

                # print(num_of_Y)
        self.PTV_scan_grid = np.array(PTV_temp)
        # calculate weight
        self.weight_matrix = []
        sobp_weight = getsobpweight(self.data, scan_depths, self.resolutionz, sigma=3)
        for scan_point in self.PTV_scan_grid:
            idx = np.squeeze(np.where(scan_depths == scan_point[2]))
            self.weight_matrix.append(sobp_weight[idx])
        self.weight_matrix = np.array(self.weight_matrix)
        return self.PTV_scan_grid, self.weight_matrix
    # def forward(self):


if __name__ == '__main__':
    myweight = weightIni()
    scan_gird, weight_matrix = myweight._weight_init()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # for point, weight in zip(scan_gird, weight_matrix):
    #     ax.scatter(point, weight)
    for point in scan_gird:
        ax.scatter(point[0], point[1], point[2], color='y')
    center = [50, 50, 100]
    radius = 8
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='b')
    plt.show()
    print('finish')
