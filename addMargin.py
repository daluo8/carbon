import os
import numpy as np
from scipy import ndimage
import pydicom
import matplotlib.pyplot as plt
import cv2
import numba as nb
import shapely.geometry as geometry


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
                                       pixel_spacing[2]) + 1)

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

    return np.array(image, dtype=np.int16)


def get_rotation_matrix(gantryAngle, couchAngle):
    # counter-clockwise around z-axis
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


def add_margin(VOIsegment, resolution, margin):
    voxel_margins = np.round(margin / resolution)
    VOIenlarged = VOIsegment.copy()
    [x_upper_lim, y_upper_lim, z_upper_lim] = VOIsegment.shape
    for cnt in range(int(max(voxel_margins))):
        # for multiple loops consider just added margin
        [xcoord, ycoord, zcoord] = np.where(VOIenlarged > 0)
        # find indices on border and take out
        borderIx = np.where(xcoord == 0) or np.where(xcoord == x_upper_lim - 1) or \
                   np.where(ycoord == 0) or np.where(ycoord == y_upper_lim - 1) or \
                   np.where(zcoord == 0) or np.where(zcoord == z_upper_lim - 1)
        xcoord[borderIx] = []
        ycoord[borderIx] = []
        zcoord[borderIx] = []

        dx = voxel_margins[0] > cnt
        dy = voxel_margins[1] > cnt
        dz = voxel_margins[2] > cnt
        arr = np.linspace(-1, 1, 3)
        for i in range(len(arr)):
            for j in range(len(arr)):
                for k in range(len(arr)):
                    if abs(arr[i]) + abs(arr[j]) + abs(arr[k]) == 0 or abs(arr[i]) + abs(arr[j]) + abs(arr[k]) > 1:
                        continue
                    xboard = np.array(xcoord + arr[i] * dx, dtype=np.int32)
                    yboard = np.array(ycoord + arr[j] * dy, dtype=np.int32)
                    zboard = np.array(zcoord + arr[k] * dz, dtype=np.int32)

                    VOIenlarged[xboard, yboard, zboard] = 1
    return VOIenlarged


def get_ct_rotm(gantryangle, couchangle, isocenter):
    rot_mat_z = cv2.getRotationMatrix2D((isocenter[0], isocenter[1]), gantryangle, scale=1)
    rot_mat_y = cv2.getRotationMatrix2D((isocenter[0], isocenter[2]), couchangle, scale=1)
    rot_mat_conz = cv2.getRotationMatrix2D((isocenter[1], isocenter[0]), -gantryangle, scale=1)
    rot_mat_cony = cv2.getRotationMatrix2D((isocenter[0], isocenter[2]), couchangle, scale=1)
    return rot_mat_z, rot_mat_y, rot_mat_conz, rot_mat_cony


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


def get_scan_grid(segmask, isocenter, ctresolution, spotspacing):
    target_points = np.where(segmask > 0)
    xcoord = (target_points[0] - isocenter[0]) * ctresolution[0]
    ycoord = (target_points[1] - isocenter[1]) * ctresolution[1]
    zcoord = (target_points[2] - isocenter[2]) * ctresolution[2]

    xspot = np.abs(xcoord) // spotspacing[0] * spotspacing[0] * np.sign(xcoord)
    yspot = np.abs(ycoord) // spotspacing[1] * spotspacing[1] * np.sign(ycoord)
    zspot = np.abs(zcoord) // spotspacing[2] * spotspacing[2] * np.sign(zcoord)
    scan_points = []

    for i in range(len(xspot)):
        if [xspot[i], yspot[i], zspot[i]] in scan_points:
            continue
        else:
            scan_points.append([xspot[i], yspot[i], zspot[i]])

    arr = np.linspace(-1, 1, 3)
    for i in range(len(arr)):
        for j in range(len(arr)):
            for k in range(len(arr)):
                if abs(arr[i]) + abs(arr[j]) + abs(arr[k]) == 0 or abs(arr[i]) + abs(arr[j]) + abs(arr[k]) > 1:
                    continue
                xspot_temp = xspot + arr[i] * spotspacing[0]
                yspot_temp = yspot + arr[j] * spotspacing[1]
                zspot_temp = zspot + arr[k] * spotspacing[2]
                for x, y, z in zip(xspot_temp, yspot_temp, zspot_temp):
                    if [x, y, z] in scan_points:
                        continue
                    else:
                        scan_points.append([x, y, z])

    #             np.array(xcoord + arr[i] * dx, dtype=np.int32)
    # for i in range(len(xspot)):
    #     if [xspot[i],yspot[i],zspot[i]] in scan_points:
    #         continue
    #     else:
    #         scan_points.append([xspot[i],yspot[i],zspot[i]])

    return np.array(scan_points)


path = 'Anonymized0706/'
patient = load_CT(path)
# ROI segmentation
segmentation, contour = load_label(path, patient)
# HU
patient_pixels = get_pixels_hu(patient)

CTV_points = np.where(segmentation > 0)

# gantryAngle, couchAngle = np.pi*3/4, 0
# rotMat = get_rotation_matrix(gantryAngle, couchAngle)

resolution = np.array(np.append(patient[0].PixelSpacing, patient[0].SliceThickness), dtype=np.float32)
origin_pos = patient[0].ImagePositionPatient
dim = patient_pixels.shape
PTVmargin = [3, 3, 3]
Isocenter_idx = np.round(np.mean(np.array(CTV_points), 1))
CTV_enlarged = add_margin(segmentation, resolution, PTVmargin)
# for slice_z in range(min(CTV_points[2]), max(CTV_points[2])):
#     segmentation[int(Isocenter_idx[0]), int(Isocenter_idx[1]), slice_z] = 2
# HU, stopping_power = HU_2_stoptable()
# patient_waterED = get_water_eq(patient_pixels[0], HU, stopping_power)
gantry_angle, couch_angle = 45, 0
ct_rotz, ct_roty, seg_rotz, seg_roty = get_ct_rotm(gantry_angle, couch_angle, Isocenter_idx)
rot_mat_segz = cv2.getRotationMatrix2D((Isocenter_idx[1], Isocenter_idx[0]), 45, scale=1)
patient_rotz = np.zeros(patient_pixels.shape)
patient_roty = np.zeros(patient_pixels.shape)
segmentation_z = np.zeros(segmentation.shape)
segmentation_y = np.zeros(segmentation.shape)
# height
for slice_z in range(min(CTV_points[2]), max(CTV_points[2]), 3):
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(patient_pixels[slice_z], cmap=plt.cm.gray)
    # idxs = np.where(CTV_points[2] == slice_z)
    # plt.scatter(CTV_points[0][idxs], CTV_points[1][idxs])
    # plt.scatter(Isocenter_idx[0],Isocenter_idx[1], marker='x')
    patient_rotz[slice_z] = cv2.warpAffine(patient_pixels[slice_z], ct_rotz, patient_pixels[slice_z].shape,
                                           borderValue=-2048)

    segmentation_z[:, :, slice_z] = cv2.warpAffine(segmentation[:, :, slice_z], seg_rotz,
                                                   patient_pixels[slice_z].shape,
                                                   borderValue=0)
#     plt.subplot(122)
#     plt.imshow(patient_rotz[slice_z], cmap=plt.cm.gray)
#     idxs=np.where(segmentation_z[:, :, slice_z]>0)
#     plt.scatter(idxs[0], idxs[1])
#     plt.scatter(Isocenter_idx[0], Isocenter_idx[1], marker='x')
# plt.show()
# spot spaing [x,y,z]
spotspacing = [3, 3, 3]
scan_points = get_scan_grid(segmentation, Isocenter_idx, resolution, spotspacing)
scan_points_z = get_scan_grid(segmentation_z, Isocenter_idx, resolution, spotspacing)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(scan_points[:, 0], scan_points[:, 1], scan_points[:, 2])

fig1 = plt.figure()
ax = fig1.add_subplot(projection='3d')
ax.scatter(scan_points_z[:, 0], scan_points_z[:, 1], scan_points_z[:, 2])
plt.show()
