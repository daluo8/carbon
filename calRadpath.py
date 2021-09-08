import os
import numpy as np
from scipy import ndimage
import pydicom
import matplotlib.pyplot as plt
import numba as nb


# import cv2
# from skimage import measure

def load_CT(path):
    CTdata = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'CT' in file:
                slice = pydicom.read_file(path + file)
                CTdata.append(slice)
    CTdata.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return CTdata


####interplote function
def interplote(points):
    added = []
    for i in range(len(points) - 1):
        dist = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
        if dist > 1.4:  # and dist <15:
            pair = [points[i], points[i + 1]]
            # interplate points according to their distance
            if np.abs(points[i][0] - points[i + 1][0]) > np.abs(points[i][1] - points[i + 1][1]):
                # x dis>y dis
                min_idx = np.argmin([points[i][0], points[i + 1][0]])
                xx = np.linspace(start=pair[min_idx][0], stop=pair[1 - min_idx][0],
                                 num=pair[1 - min_idx][0] - pair[min_idx][0] + 2, dtype='int32')
                interp = np.interp(xx, [pair[min_idx][0], pair[1 - min_idx][0]],
                                   [pair[min_idx][1], pair[1 - min_idx][1]])
                for dummy in zip(xx, interp):
                    added.append([int(dummy[0]), int(dummy[1])])

            else:  # y dis>=x dis
                min_idx = np.argmin([points[i][1], points[i + 1][1]])
                yy = np.linspace(start=pair[min_idx][1], stop=pair[1 - min_idx][1],
                                 num=pair[1 - min_idx][1] - pair[min_idx][1] + 2, dtype='int32')
                interp = np.interp(yy, [pair[min_idx][1], pair[1 - min_idx][1]],
                                   [pair[min_idx][0], pair[1 - min_idx][0]])
                for dummy in zip(interp, yy):
                    added.append([int(dummy[0]), int(dummy[1])])

    return [list(x) for x in set(tuple(x) for x in added + points)]


##Read  RS label###
def load_label(path, CTdata):
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'RS' in file:
                structure = pydicom.read_file(path + file, force=True)
    shape = [CTdata[0].Columns, CTdata[0].Rows, np.array(CTdata).shape[0]]
    origin = CTdata[0].ImagePositionPatient
    pixel_spacing = np.array(np.append(CTdata[0].PixelSpacing, CTdata[0].SliceThickness), dtype=np.float32)
    index_CTVnx = 0
    flag_CTVnx = 0
    orientation = CTdata[0].ImageOrientationPatient
    m = np.matrix(
        [[orientation[0], orientation[3], 0, origin[0]],
         [orientation[1], orientation[4], 0, origin[1]],
         [orientation[2], orientation[5], 0, origin[2]],
         [0, 0, 0, 1]])
    m1 = np.matrix(
        [[0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1]])
    rot_m = m * m1
    for item in structure.StructureSetROISequence:
        # if 'CTV' in item.ROIName:
        if item.ROIName == 'CTV2':
            flag_CTVnx = 1
            index_CTVnx = structure.StructureSetROISequence.index(item)
            break
    contour_nx = np.zeros(shape)
    segmentation_nx = np.zeros(shape)
    if flag_CTVnx == 1:
        roi_nx = structure.ROIContourSequence[index_CTVnx]  # 靶区对应的ROI contour信息
        if hasattr(roi_nx, 'ContourSequence'):  # 判断roi是不是含有contoursequence
            roi_number_nx = len(roi_nx.ContourSequence._list)  # 如果有的话，统计靶区对应的切片数量
            for ii in range(roi_number_nx):
                plane_contour_nx = roi_nx.ContourSequence[ii]
                contour_points_nx = zip(*[iter(plane_contour_nx.ContourData)] * 3)
                contour_points_nx = list(contour_points_nx)
                z_voxel_nx = int(round((contour_points_nx[0][2] - origin[2]) /
                                       pixel_spacing[2]))

                test_aa = []
                for point in contour_points_nx:
                    x = rot_m * np.matrix([[float(point[0])], [float(point[1])], [0], [1]])
                    y = rot_m * np.matrix([[float(point[0])], [float(point[1])], [0], [1]])
                    x_voxel = int(
                        round((float(x[0]) - origin[0]) / pixel_spacing[0])) + int(shape[0] / 2)
                    y_voxel = int(
                        round((float(y[1]) - origin[1]) / pixel_spacing[1])) + int(shape[1] / 2)
                    test_aa.append([x_voxel, y_voxel])
                # contour_points_nx=np.array(contour_points_nx)
                test_aa.append(test_aa[0])
                temp_contour_nx = interplote(test_aa)
                temp_contour_nx = np.array(temp_contour_nx)
                contour_nx[temp_contour_nx[:, 1], temp_contour_nx[:,
                                                  0], z_voxel_nx] = 1  # mind the dimension matching
                seg_nx = ndimage.binary_fill_holes(
                    contour_nx[:, :, z_voxel_nx])  # fill the inside of the contour
                segmentation_nx[:, :, z_voxel_nx] = seg_nx
        segmentation = segmentation_nx
        return segmentation.astype(np.uint8), contour_nx.astype(np.uint8)


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
    # image.reshape((image[0].shape[0],image[0].shape[1],-1))
    return np.array(image, dtype=np.int16)


def get_rotation_matrix(gantryAngle, couchAngle):
    #counter-clockwise around z-axis
    m_gantry = np.matrix(
        [[np.cos(gantryAngle), -np.sin(gantryAngle), 0],
         [np.sin(gantryAngle), np.cos(gantryAngle), 0],
         [0, 0, 1]])
    # counter-clockwise around y-axis
    m_couch = np.matrix(
        [[np.cos(couchAngle), 0, np.sin(gantryAngle)],
         [0, 1, 0],
         [-np.sin(couchAngle), 0, np.cos(couchAngle)]])
    rotMat = m_gantry * m_couch
    return rotMat


def HU_2_stoptable():
    # HU to Stoppingpower table ref to matRad
    HU = [-1024, 200, 449, 2000, 2048, 3071]
    StoppingPower = [0.00324, 1.2, 1.20001, 2.49066, 2.53060, 2.53061]
    return HU, StoppingPower


def get_water_eq(ctcube, HU, StoppingPower):
    water_eq_cube = np.zeros(ctcube.shape)

    # @nb.jit(nopython=True)
    def HU2stoppingpower(x):
        return np.interp(x, HU, StoppingPower)

    function_vector = np.vectorize(HU2stoppingpower)
    for slice_number in range(len(ctcube)):
        water_eq_cube[slice_number] = function_vector(ctcube[slice_number])
    return water_eq_cube


def cal_rad_depth(origin, isocenter, resolution, source_point, target_point, cubes):
    source_point = source_point + isocenter
    target_point = target_point + isocenter
    # Save the numbers of planes.
    z_num_planes, y_num_planes, x_num_planes = cubes.shape
    x_num_planes = x_num_planes + 1
    y_num_planes = y_num_planes + 1
    z_num_planes = z_num_planes + 1
    # eq 11
    # Calculate the distance from source to target point.
    d12 = np.linalg.norm(np.array(source_point) - np.array(target_point))
    # eq 3
    # Position of first planes in millimeter. 0.5 because the central position
    if origin[0]>0:
        xPlane_1 = .5 * resolution[0]
        xPlane_end = (x_num_planes - .5) * resolution[0]
    else:
        xPlane_1 = -(x_num_planes - .5) * resolution[0]
        xPlane_end = -.5 * resolution[0]
    if origin[1]>0:
        yPlane_1 = .5 * resolution[1]
        yPlane_end = (y_num_planes - .5) * resolution[1]
    else:
        yPlane_1 = -(y_num_planes - .5) * resolution[1]
        yPlane_end = -.5 * resolution[1]
    zPlane_1 = .5 * resolution[2]
    zPlane_end = (z_num_planes - .5) * resolution[2]

    # eq 4
    # Calculate parametrics values of \alpha_{min} and \alpha_{max} for every
    # axis, intersecting the ray with the sides of the CT.
    if target_point[0] != source_point[0]:
        aX_1 = (xPlane_1 - source_point[0]) / (target_point[0] - source_point[0])
        aX_end = (xPlane_end - source_point[0]) / (target_point[0] - source_point[0])
    else:
        aX_1 = -1
        aX_end = 2
    if target_point[1] != source_point[1]:
        aY_1 = (yPlane_1 - source_point[1]) / (target_point[1] - source_point[1])
        aY_end = (yPlane_end - source_point[1]) / (target_point[1] - source_point[1])
    else:
        aY_1 = -1
        aY_end = 2
    if target_point[2] != source_point[2]:
        aZ_1 = (zPlane_1 - source_point[2]) / (target_point[2] - source_point[2])
        aZ_end = (zPlane_end - source_point[2]) / (target_point[2] - source_point[2])
    else:
        aZ_1 = -1
        aZ_end = 2
    # eq 5
    # Compute the \alpha_{min} and \alpha_{max} in terms of parametric values
    # given by equation 4.
    alpha_min = max((0, min(aX_1, aX_end), min(aY_1, aY_end), min(aZ_1, aZ_end)))
    alpha_max = min((1, max(aX_1, aX_end), max(aY_1, aY_end), max(aZ_1, aZ_end)))
    # eq 6
    # Calculate the range of indeces who gives parametric values for
    # intersected planes.
    if target_point[0] == source_point[0]:
        i_min = []
        i_max = []
    elif target_point[0] > source_point[0]:
        i_min = x_num_planes - (xPlane_end - alpha_min * (target_point[0] - source_point[0]) - source_point[0]) / \
                resolution[0]
        i_max = 1 + (source_point[0] + alpha_max * (target_point[0] - source_point[0]) - xPlane_1) / resolution[0]
        i_min = np.ceil(1 / 1000 * (round(1000 * i_min)))
        i_max = np.floor(1 / 1000 * (round(1000 * i_max)))
    else:
        i_min = x_num_planes - (
                xPlane_end - alpha_max * (target_point[0] - source_point[0]) - source_point[0]) / resolution[0]
        i_max = 1 + (source_point[0] + alpha_min * (target_point[0] - source_point[0]) - xPlane_1) / resolution[0]
        i_min = np.floor(1 / 1000 * (round(1000 * i_min)))
        i_max = np.floor(1 / 1000 * (round(1000 * i_max)))
    if target_point[1] == source_point[1]:
        j_min = []
        j_max = []
    elif target_point[1] > source_point[1]:
        j_min = y_num_planes - (yPlane_end - alpha_min * (target_point[1] - source_point[1]) - source_point[1]) / \
                resolution[1]
        j_max = 1 + (source_point[1] + alpha_max * (target_point[1] - source_point[1]) - yPlane_1) / resolution[1]
        j_min = np.ceil(1 / 1000 * (round(1000 * j_min)))
        j_max = np.floor(1 / 1000 * (round(1000 * j_max)))
    else:
        j_min = y_num_planes - (
                yPlane_end - alpha_max * (target_point[1] - source_point[1]) - source_point[1]) / resolution[1]
        j_max = 1 + (source_point[1] + alpha_min * (target_point[1] - source_point[1]) - yPlane_1) / resolution[1]
        j_min = np.floor(1 / 1000 * (round(1000 * j_min)))
        j_max = np.floor(1 / 1000 * (round(1000 * j_max)))
    if target_point[2] == source_point[2]:
        k_min = []
        k_max = []
    elif target_point[2] > source_point[2]:
        k_min = x_num_planes - (xPlane_end - alpha_min * (target_point[2] - source_point[2]) - source_point[2]) / \
                resolution[2]
        k_max = 1 + (source_point[2] + alpha_max * (target_point[2] - source_point[2]) - zPlane_1) / resolution[2]
        k_min = np.ceil(1 / 1000 * (round(1000 * k_min)))
        k_max = np.floor(1 / 1000 * (round(1000 * k_max)))
    else:
        k_min = x_num_planes - (
                xPlane_end - alpha_max * (target_point[2] - source_point[2]) - source_point[2]) / resolution[2]
        k_max = 1 + (source_point[2] + alpha_min * (target_point[2] - source_point[2]) - zPlane_1) / resolution[2]
        k_min = np.floor(1 / 1000 * (round(1000 * k_min)))
        k_max = np.floor(1 / 1000 * (round(1000 * k_max)))
    # eq 7
    # For the given range of indices, calculate the paremetrics values who
    # represents intersections of the ray with the plane.
    if i_min != i_max:
        if target_point[0] > source_point[0]:
            alpha_x = (resolution[0] * np.linspace(i_min, i_max, abs(i_max - i_min) + 1) - source_point[0] - .5 * resolution[
                0]) / (target_point[0] - source_point[0])
        else:
            alpha_x = (resolution[0] * np.linspace(i_max, i_min, abs(i_max - i_min) + 1) - source_point[0] - .5 * resolution[
                0]) / (target_point[0] - source_point[0])
    else:
        alpha_x = []
    if j_min != j_max:
        if target_point[1] > source_point[1]:
            alpha_y = (resolution[1] * np.linspace(j_min, j_max, abs(j_max - j_min) + 1) - source_point[1] - .5 * resolution[
                1]) / (target_point[1] - source_point[1])
        else:
            alpha_y = (resolution[1] * np.linspace(j_max, i_min, abs(i_max - i_min) + 1) - source_point[1] - .5 * resolution[
                1]) / (target_point[1] - source_point[1])
    else:
        alpha_y = []
    if k_min != k_max:
        if target_point[2] > source_point[2]:
            alpha_z = (resolution[2] * np.linspace(k_min, k_max, abs(k_max - k_min) + 1) - source_point[2] - .5 *
                       resolution[
                           2]) / (target_point[2] - source_point[2])
        else:
            alpha_z = (resolution[2] * np.linspace(k_max, k_min, abs(k_max - k_min) + 1) - source_point[2] - .5 *
                       resolution[
                           2]) / (target_point[2] - source_point[2])
    else:
        alpha_z = []
    # eq 8
    # Merge parametrics sets
    alphas = [alpha_min] + list(alpha_x) + list(alpha_y) + list(alpha_z) + [alpha_max]
    alphas = np.unique(alphas)
    # eq 10
    # Calculate the voxel intersection length.
    l = d12 * np.diff(alphas)
    # eq 13
    # Calculate \alpha_{middle}
    alphas_mid = .5 * (alphas[:-1] + alphas[1:])
    # eq 12
    # Calculate the voxel indices: first convert to physical coords
    i_mm = source_point[0] + alphas_mid * (target_point[0] - source_point[0])
    j_mm = source_point[1] + alphas_mid * (target_point[1] - source_point[1])
    k_mm = source_point[2] + alphas_mid * (target_point[2] - source_point[2])
    # then convert to voxel index
    i = np.round(i_mm / resolution[0])
    j = np.round(j_mm / resolution[1])
    k = np.round(k_mm / resolution[2])
    # Handle numerical instabilities at the borders.
    i[i < 0] = 0
    j[j < 0] = 0
    k[k < 0] = 0
    i[i >= x_num_planes - 1] = x_num_planes - 2
    j[j >= y_num_planes - 1] = y_num_planes - 2
    k[k >= z_num_planes - 1] = z_num_planes - 2

    # convert to linear indices
    # ix = k + (j - 1) * cubes[0].shape[0] + (i - 1) * cubes[0].shape[0] * cubes[0].shape[1]
    # obtains the values from cubes
    rho = []
    i, j, k = i.astype(int), j.astype(int), k.astype(int)
    for idx in range(len(i)):
        rho.append(cubes[k[idx], i[idx], j[idx]])
    rho = np.array(rho)
    rad_depth = np.dot(l, rho)
    return alphas, l, rho, d12, rad_depth


path = 'Anonymized0706/'
patient = load_CT(path)
segmentation, contour = load_label(path, patient)
patient_pixels = get_pixels_hu(patient)
HU, stopping_power = HU_2_stoptable()

PTV_points = np.where(segmentation > 0)
patient_waterED = get_water_eq(patient_pixels[min(PTV_points[2]): max(PTV_points[2])], HU, stopping_power)
gantryAngle, couchAngle = np.pi*1/4, 0
rotMat = get_rotation_matrix(gantryAngle, couchAngle)

resolution = np.array(np.append(patient[0].PixelSpacing, patient[0].SliceThickness), dtype=np.float32)
origin_pos = patient[0].ImagePositionPatient
dim=patient_waterED.shape
Isocenter_idx = np.round(np.mean(np.array(PTV_points), 1))
Isocenter_idx_cen = Isocenter_idx-[dim[1]/2, dim[2]/2, 0]
Isocenter = origin_pos + Isocenter_idx_cen*resolution
source_pos = np.array([0, -1000, 0])
source_pos = np.squeeze(np.array(rotMat * source_pos.reshape(-1, 1)))
target_pos = [0, 0, 0]
alphas, l, rho, d12, rad_depth = cal_rad_depth(origin_pos, Isocenter, resolution, source_pos, target_pos, patient_waterED)
point1 = source_pos + Isocenter
point2 = target_pos + Isocenter
point1_idx = (point1-origin_pos) / resolution+[dim[1]/2, dim[2]/2, 0]
point2_idx = (point2-origin_pos) / resolution+[dim[1]/2, dim[2]/2, 0]
for z_nx in range(min(PTV_points[2]), max(PTV_points[2]), 1):
    plt.figure()
    plt.title(z_nx)
    # patient_watereq = get_water_eq(patient_pixels[z_nx], HU, StoppingPower)
    plt.imshow(patient_pixels[z_nx], cmap=plt.cm.gray)
    idxs = np.where(PTV_points[2] == z_nx)
    plt.scatter(PTV_points[0][idxs], PTV_points[1][idxs], z_nx)
    plt.plot([point1_idx[0], point2_idx[0]], [point1_idx[1], point2_idx[1]], 'r')
plt.show()
