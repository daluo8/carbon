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
        if dist > 1.4 :#and dist <15:
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
    rot_m=m*m1
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

    return np.array(image, dtype=np.int16)

def get_rotation_matrix(gantryAngle,couchAngle):
    m_gantry=np.matrix(
        [[cos(gantryAngle),-sin(gantryAngle),0],
         [sin(gantryAngle),cos(gantryAngle),0],
         [0, 0, 1]])
    return m_gantry
path='Anonymized0706/'
patient = load_CT(path)
segmentation, contour = load_label(path, patient)
patient_pixels = get_pixels_hu(patient)
PTV_points = np.where(segmentation>0)
gantryAngle,couchAngle=0,0
# rotMat=get_rotation_matrix(gantryAngle,couchAngle)

# Isocenter=np.round(np.mean(np.array(PTV_points), 1))
for z_nx in range(min(PTV_points[2]), max(PTV_points[2]), 1):
    plt.figure()
    plt.title(z_nx)
    # patient_watereq = get_water_eq(patient_pixels[z_nx], HU, StoppingPower)
    plt.imshow(patient_pixels[z_nx], cmap=plt.cm.gray)
    idxs = np.where(PTV_points[2] == z_nx)
    plt.scatter(PTV_points[0][idxs], PTV_points[1][idxs], z_nx)
plt.show()