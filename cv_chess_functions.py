from audioop import mul
from configparser import ExtendedInterpolation
import math
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean
import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image
import re
import glob
import PIL
import os
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, Sequence
from calibration import get_calibration_data
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import copy
import requests
def linear_func(x1,y1,x2,y2):
    a = (y2-y1)/(x2-x1)
    b=y1-a*x1
    return {'a':a,'b':b}
# 線分ABと線分CDの交点を求める関数
def _calc_cross_point(pointA, pointB, pointC, pointD):
    cross_point = (0,0)
    bunbo = (pointB[0] - pointA[0]) * (pointD[1] - pointC[1]) - (pointB[1] - pointA[1]) * (pointD[0] - pointC[0])

    # 直線が平行な場合
    if (bunbo == 0):
        return False, cross_point

    vectorAC = ((pointC[0] - pointA[0]), (pointC[1] - pointA[1]))
    r = ((pointD[1] - pointC[1]) * vectorAC[0] - (pointD[0] - pointC[0]) * vectorAC[1]) / bunbo
    s = ((pointB[1] - pointA[1]) * vectorAC[0] - (pointB[0] - pointA[0]) * vectorAC[1]) / bunbo

    # rを使った計算の場合
    distance = ((pointB[0] - pointA[0]) * r, (pointB[1] - pointA[1]) * r)
    cross_point = (int(pointA[0] + distance[0]), int(pointA[1] + distance[1]))

    # sを使った計算の場合
    # distance = ((pointD[0] - pointC[0]) * s, (pointD[1] - pointC[1]) * s)
    # cross_point = (int(pointC[0] + distance[0]), int(pointC[1] + distance[1]))

    return True, cross_point
# points=(p1,p2,p3,p4)
# ４の点からチェス盤のマス目のポイントを得る関数
def points_by_points(points,X=9):
    print(points)
    if points[2][0]>points[3][0]:
        points[2],points[3]=points[3],points[2]
    # t_l=(math.fabs(points[0][0]-points[1][0])**2+math.fabs(points[0][1]-points[1][1])**2)**0.5
    # b_l=(math.fabs(points[2][0]-points[3][0])**2+math.fabs(points[2][1]-points[3][1])**2)**0.5
    # height = (math.fabs(points[0][0]-points[2][0])**2+math.fabs(points[1][1]-points[3][1])**2)**0.5
    # print('t_l',t_l)
    # print('b_l',b_l)
    # print('height',height)
    # h_a=t_l+b_l/2
    # s=(t_l/b_l)/height
    #print('s',s)
    h_t_lines = np.column_stack([np.linspace(points[0][0], points[1][0], X),np.linspace(points[0][1], points[1][1], X)])
    h_b_lines = np.column_stack([np.linspace(points[2][0], points[3][0], X),np.linspace(points[2][1], points[3][1], X)])
    h_b_lines=np.sort(h_b_lines,axis=0)
    retpoints=[]
    b=math.fabs(points[0][0]-points[1][0])/math.fabs(points[2][0]-points[3][0])
    for i in range(0,len(h_t_lines)):
        print(h_t_lines[i][0], h_b_lines[i][0])
        _points = np.column_stack([np.linspace(h_t_lines[i][0], h_b_lines[i][0], X),np.linspace(h_t_lines[i][1], h_b_lines[i][1], X)])
        retpoints.append(_points)
    print(retpoints)
    return retpoints

# 角ABCを計算(rad)
def get_angle(_pointA, _pointB, _pointC):
    pointA, pointB, pointC = np.array(_pointA,dtype=float), np.array(_pointB,dtype=float), np.array(_pointC,dtype=float)
    vec_BA = pointA - pointB
    vec_BC = pointC - pointB
    cos_ABC = np.inner(vec_BA, vec_BC) / (np.linalg.norm(vec_BA) * np.linalg.norm(vec_BC))
    return np.arccos(cos_ABC)

# AをOを中心に (反時計回りに)angle度回転
def rotate(_pointA, _pointO, angle):
    pointA, pointO = np.array(_pointA,dtype=float), np.array(_pointO,dtype=float)
    pointA -= pointO
    pointA = np.array([pointA[0] * np.cos(angle) - pointA[1] * np.sin(angle), pointA[0] * np.sin(angle) + pointA[1] * np.cos(angle)])
    pointA += pointO
    return pointA

# 一辺がA-M-B, もう一辺がC-D と並んでいることを仮定 
def partition_by_ratio(_pointA, _pointB, _pointC, _pointD, _pointM, X=9):
    pointA, pointB, pointC, pointD, pointM = np.array(_pointA,dtype=float), np.array(_pointB,dtype=float), np.array(_pointC,dtype=float), np.array(_pointD,dtype=float), np.array(_pointM,dtype=float)
    #infinity_point = np.array(_calc_cross_point(pointA, pointB, pointC, pointD)[1],dtype=float)

    ratio = np.power(np.linalg.norm(pointB - pointM) / np.linalg.norm(pointM - pointA), 0.25)
    initial = 1.0 * (1.0 - ratio) / (1.0 - np.power(ratio, X - 1))
    print(pointA,pointB,pointC,pointD,pointM,pointB-pointM,pointM-pointA)
    print(initial, ratio)
    ret_lines = np.zeros([9,2,2],dtype=float)
    vec_sum = 0
    vec = (pointB - pointA) * initial
    for i in range(X):
        ret_lines[i][0] = pointA + vec_sum
        vec_sum += vec
        vec *= ratio

    vec_sum = 0
    vec = (pointD - pointC) * initial

    for i in range(X):
        ret_lines[i][1] = pointC + vec_sum
        vec_sum += vec
        vec *= ratio

    print(ret_lines)
    return ret_lines

# points=(p1,p2,p3,p4)
# ４の点からチェス盤のマス目のポイントを得る関数(二点透視図法による)
# points[0],points[1],points[2]が一列にならんでおり、points[0],points[3],points[4]が一列にならんでおり、points[5]が残りの角である
def points_by_points_with_twopoint_perspective(points,X=9):
    print(points)
    if points[4][0]>points[5][0]:
        points[4],points[5]=points[5],points[4]
    
    vertical_lines=partition_by_ratio(points[0],points[2],points[4],points[5],points[1],X)
    horizontal_lines=partition_by_ratio(points[0],points[4],points[2],points[5],points[3],X)
    ret_points = np.zeros([9,9,2])
    for i in range(X):
        for j in range(X):
            ret_points[i][j] = np.array(_calc_cross_point(horizontal_lines[i][0], horizontal_lines[i][1], vertical_lines[j][0], vertical_lines[j][1])[1])
    print(ret_points)
    return ret_points

# ４の点からチェス盤のマス目のポイントを得る関数(透視投影行列による)
def points_by_points_with_perspective_projection(points,P=9):
    print(points)
    if points[2][0]>points[3][0]:
        points[2],points[3]=points[3],points[2]
    matrix_size = 12
    coefficient_matrix = np.zeros([matrix_size,matrix_size],dtype=float)
    extended_matrix = np.zeros([matrix_size,1],dtype=float)
    for i in range(2):
        for j in range(2):
            index = i * 2 + j
            x = points[index][0]
            y = points[index][1]
            X = i * 9.0
            Y = j * 9.0
            Z = 1.0
            print(i,j,index,x,y,X,Y,Z)
            coefficient_matrix[index * 3][0] = X
            coefficient_matrix[index * 3][1] = Y
            coefficient_matrix[index * 3][2] = Z
            coefficient_matrix[index * 3][3] = 1
            extended_matrix[index * 3] = x
            coefficient_matrix[index * 3 + 1][4] = X
            coefficient_matrix[index * 3 + 1][5] = Y
            coefficient_matrix[index * 3 + 1][6] = Z
            coefficient_matrix[index * 3 + 1][7] = 1
            extended_matrix[index * 3 + 1] = y
            coefficient_matrix[index * 3 + 2][8] = X
            coefficient_matrix[index * 3 + 2][9] = Y
            coefficient_matrix[index * 3 + 2][10] = Z
            coefficient_matrix[index * 3 + 2][11] = 1
            extended_matrix[index * 3 + 2] = 1
    for i in range(12):
        coefficient_matrix[i][i] += 0.000000001
    print(coefficient_matrix)
    print(extended_matrix)
    answer = np.linalg.solve(coefficient_matrix, extended_matrix)
    print(answer)
    print(answer.shape)
    perspective_projection_matrix = np.reshape(answer,[3, 4])
    print(perspective_projection_matrix)
    ret_points = []
    for i in range(P):
        for j in range(P):
            X = i * 9.0
            Y = j * 9.0
            Z = 1.0
            point = perspective_projection_matrix * np.matrix([[X], [Y], [Z], [1]])
            print(point)
            ret_points.append(np.array([float(point[0][0]), float(point[1][0])], dtype=float))
            print(float(point[0][0]))
    print(ret_points)
    return ret_points

# Read image and do lite image processing
def read_img(file):
    img = cv2.imread(str(file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))
    return img, gray_blur


# Canny edge detection
def canny_edge(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges
    # return cv2.Canny(img,50,150,apertureSize = 3)


# Hough line detection
def hough_line(edges, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)
    if lines is None:
      return None
    lines = np.reshape(lines, (-1, 2))
    return lines


# Separate line into horizontal and vertical
def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines


# Find the intersections of the lines
def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)


# Hierarchical cluster (by euclidean distance) intersection points
def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])


# Average the y value in each row and augment original points
def augment_points(points):
    points_shape = list(np.shape(points))
    augmented_points = []
    for row in range(int(points_shape[0] / 11)):
        start = row * 11
        end = (row * 11) + 10
        rw_points = points[start:end + 1]
        rw_y = []
        rw_x = []
        for point in rw_points:
            x, y = point
            rw_y.append(y)
            rw_x.append(x)
        y_mean = mean(rw_y)
        for i in range(len(rw_x)):
            point = (rw_x[i], y_mean)
            augmented_points.append(point)
    augmented_points = sorted(augmented_points, key=lambda k: [k[1], k[0]])
    return augmented_points


# Crop board into separate images
def write_crop_images(img, points, img_count, folder_path='./raw_data/',X=9,up_extend=0.1,down_extend=0.1,right_extend=0.1,left_extend=0.1):
                num_list = []
                shape = list(np.shape(points))
                start_point = shape[0]
                print('start_point',start_point)
                # if int(shape[0] / X) >= 8:
                # 				range_num = 8
                # else:
                # 				range_num = int((shape[0] / X) - 2)

                # for row in range(range_num):
                # 				start = start_point - (row * X)
                # 				end = (start_point - 8) - (row * X)
                # 				num_list.append(range(start, end, -1))


                for rowi in range(len(points)-1):
                    row_upper = points[rowi]
                    row_lower = points[rowi + 1]
                    #print('row',row)
                    for s in range(len(row_lower)-1):
                        # ratio_w = 1
                        #print('s',s)
                        #print('points',len(row),row[s])
                        '''
                        base_len = math.dist(row[s], row[s + 1])
                        bot_left, bot_right = row[s], row[s + 1]
                        hoo = np.array(bot_right) - np.array( bot_left)
                        if hoo[0]<=0 or hoo[1]<=0:
                            continue
                        '''
                        #start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
                        #end_x, end_y = int(bot_right[0]), int(bot_right[1])
                        start_x = int(min(row_lower[s][0], row_lower[s + 1][0], row_upper[s][0], row_upper[s + 1][0]))
                        end_x = int(max(row_lower[s][0], row_lower[s + 1][0], row_upper[s][0], row_upper[s + 1][0]))
                        start_y = int(min(row_lower[s][1], row_lower[s + 1][1], row_upper[s][1], row_upper[s + 1][1]))
                        end_y = int(max(row_lower[s][1], row_lower[s + 1][1], row_upper[s][1], row_upper[s + 1][1]))
                        dx = end_x - start_x
                        dy = end_y - start_y
                        start_x = int(max(0, start_x - dx * left_extend))
                        end_x = int(min(img.shape[1], end_x + dx * right_extend))
                        start_y = int(max(0, start_y - dy * up_extend))
                        end_y = int(min(img.shape[0], end_y + dy * down_extend))
                        #print('start_y, end_y, start_x, end_x',start_y, end_y, start_x, end_x)
                        #print(np.shape(img))
                        cropped = img[start_y: end_y, start_x: end_x]
                        #print('np.shape(cropped)',np.shape(cropped))
                        if np.shape(cropped)[1]<=1:
                            continue
                        img_count += 1
                        #print("cropped",cropped,len(cropped),np.shape(cropped))
                        os.makedirs(folder_path, exist_ok=True)
                        cv2.imwrite(folder_path + 'crop_data_image' + str(img_count) + '.jpeg', cropped)
                        #cv2.imwrite(folder_path + str(img_count) + '.jpeg', cropped)
                        #cv2.imwrite('./test_data/crop_data_image' + str(img_count) + '.jpeg', cropped)
                        #print(folder_path + 'data' + str(img_count) + '.jpeg')
                return img_count


# Crop board into separate images and shows
def x_crop_images(img, points):
    num_list = []
    img_list = []
    shape = list(np.shape(points))
    start_point = shape[0] - 14

    if int(shape[0] / 11) >= 8:
        range_num = 8
    else:
        range_num = int((shape[0] / 11) - 2)

    for row in range(range_num):
        start = start_point - (row * 11)
        end = (start_point - 8) - (row * 11)
        num_list.append(range(start, end, -1))

    for row in num_list:
        for s in row:
            base_len = math.dist(points[s], points[s + 1])
            bot_left, bot_right = points[s], points[s + 1]
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            img_list.append(cropped)
            # print(folder_path + 'data' + str(img_count) + '.jpeg')
    return img_list

def rescale(img):
    W = 1000
    height, width, depth = img.shape
    #imgScale = width 
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))

# Convert image from RGB to BGR
def convert_image_to_bgr_numpy_array(image_path, size=(224, 224)):
    image = PIL.Image.open(image_path).resize(size)
    img_data = np.array(image.getdata(), np.float32).reshape(*size, -1)
    # swap R and B channels
    img_data = np.flip(img_data, axis=2)
    return img_data



# Adjust image into (1, 224, 224, 3)
def prepare_image(image_path):
    im = convert_image_to_bgr_numpy_array(image_path)

    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    im = np.expand_dims(im, axis=0)
    return im


# Changes digits in text to ints
def atoi(text):
    return int(text) if text.isdigit() else text


# Finds the digits in a string
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


# Reads in the cropped images to a list
def grab_cell_files(folder_name='./raw_data/*'):
    img_filename_list = []
    for path_name in glob.glob(folder_name):
        img_filename_list.append(path_name)
    # img_filename_list = img_filename_list.sort(key=natural_keys)
    return img_filename_list




def classify_cells_pre_keras(model, img_filename_list): 
    img_width, img_height = 224, 224
    pred_list = []
    category_reference = {0: '1', 1: 'K'}
    
    for image_name in img_filename_list:
            img_predict = []
            img = Image.open(image_name)
            img = img.convert("RGB")
            img = img.resize((img_width, img_height))
            img_array = np.array(img, dtype=float)
            img_predict.append(img_array)
            for i in range(len(img_predict)):
                img_predict[i] *= 1./255
            img_predict = np.array(img_predict, dtype=float)
            out = model.predict(img_predict)
            print(out)
            top_pred = np.argmax(out)
            pred = category_reference[top_pred]
            pred_list.append(pred)
    '''
    img_predict = []
    for image_name in img_filename_list:
            img = Image.open(image_name)
            img = img.convert("RGB")
            img = img.resize((img_width, img_height))
            img_array = np.asarray(img, dtype=float)
            for i in range(len(img_array)):
                img_array[i] *= 1./255
            img_predict.append(img_array)
            
    img_predict = np.asarray(img_predict)
    out = model.predict(img_predict)
    for result in out:
            top_pred = np.argmax(result)
            pred = category_reference[top_pred]
            pred_list.append(pred)
    '''
    fen = ''.join(pred_list)
    fen = fen[::-1]
    fen = '/'.join(fen[i:i + 8] for i in range(0, len(fen), 8))
    sum_digits = 0
    for i, p in enumerate(fen):
        if p.isdigit():
            sum_digits += 1
        elif p.isdigit() is False and (fen[i - 1].isdigit() or i == len(fen)):
            fen = fen[:(i - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1)) + fen[i:]
            sum_digits = 0
    if sum_digits > 1:
        fen = fen[:(len(fen) - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1))
    fen = fen.replace('D', '')
    return fen

# Classifies each square and outputs the list in Forsyth-Edwards Notation (FEN)
def classify_cells(model, img_filename_list):
    category_reference = {0: '1', 1: 'K'}
    pred_list = []
    for filename in img_filename_list:
        img = prepare_image(filename)
        img *= 1./255
        #print(img)
        out = model.predict(img)
        top_pred = np.argmax(out)
        pred = category_reference[top_pred]
        pred_list.append(pred)

    fen = ''.join(pred_list)
    fen = fen[::-1]
    fen = '/'.join(fen[i:i + 8] for i in range(0, len(fen), 8))
    sum_digits = 0
    for i, p in enumerate(fen):
        if p.isdigit():
            sum_digits += 1
        elif p.isdigit() is False and (fen[i - 1].isdigit() or i == len(fen)):
            fen = fen[:(i - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1)) + fen[i:]
            sum_digits = 0
    if sum_digits > 1:
        fen = fen[:(len(fen) - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1))
    fen = fen.replace('D', '')
    return fen

# Classifies each square and outputs the list in Forsyth-Edwards Notation (FEN)
def classify_cells_to_array(model, img_filename_list):
    pred_list = []
    is_exist = np.zeros((8,8), dtype=bool)
    for filename in img_filename_list:
        img = prepare_image(filename)
        img *= 1./255
        #print(img)
        #out=np.array((0,0))
        out = model.predict(img,verbose=0)
        top_pred = np.argmax(out)
        pred_list.append(top_pred)
        #os.makedirs('./predicted_data', exist_ok=True)
        #print('./predicted_data/' + filename[-7:-5] + "-" + str(top_pred) + '.jpeg')
        #cv2.imwrite('./predicted_data/' + filename[-7:-5] + "-" + str(top_pred) + '.jpeg', cv2.imread(filename))
        #print(out)

    for i, val in enumerate(pred_list):
        if val == 1:
            is_exist[i // 8][i % 8] = True
    return is_exist
class ImageSequence(Sequence):

    def __init__(self, image_set, batch_size):
        self.set = image_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.set) / self.batch_size)

    def __getitem__(self, idx):
        batch_image = self.set[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return batch_image
# Classifies each square and outputs the list in Forsyth-Edwards Notation (FEN)
def classify_cells_to_array_simul_multi(model, img_filename_list,executor):
    pred_list = []
    is_exist = np.zeros((8,8), dtype=bool)
    #img_list = np.zeros((1024,224,224,3),dtype=float)
    img_list = np.zeros((64,224,224,3),dtype=float)
    index = 0
    outs = []
    #print(np.shape(img_list))
    imgs = []
    start=time.time()
    '''
    for filename in img_filename_list:
        #imgs.append(PIL.Image.open(filename).resize((224,224)))
        stap=time.time()
        img = prepare_image(filename)
        print('append:',(time.time()-stap)*64)
        img *= 1./255
        #print(np.shape(img))
        img_list[index] = img[0]
        #img_list=np.append(img_list, img,axis=0)
        #print(np.shape(img_list[index]))
        #outs.append(model(img,training=False))
        index += 1
        #print(np.shape(img_list))
        #print(img_list)
    '''
    predict_index=0
    for img in list(executor.map(prepare_image, img_filename_list)):
        img *= 1./255
        img_list[predict_index] = img[0]
        predict_index += 1
    #for i in range(64,1024):
        #img_list[i] = img_list[i % 64]
    end=time.time()
    
    images = ImageSequence(img_list,64)

    print("image_prepare_time:",end-start)
    #print(img_list)
    #print(np.shape(img_list))
    #gen = tf.keras.preprocessing.image.ImageDataGenerator()
    #gen = gen.flow(img_list,batch_size=64)
    start=time.time()
    #outs = model.predict(images,verbose=1,batch_size=16,workers=4,use_multiprocessing=False)
    #outs = model.predict(images,verbose=1,workers=4,use_multiprocessing=True)
    outs = model.predict(images,verbose=1,max_queue_size=16,workers=4,use_multiprocessing=False)
    #outs = model.predict_generator(gen,verbose=1,workers=16,use_multiprocessing=True)
    end=time.time()
    #print("prediction_time:",end-start)
    #outs = model(img_list,training=False)
    for out in outs:
        top_pred = np.argmax(out)
        pred_list.append(top_pred)
    #for i, val in enumerate(pred_list):
        #if val == 1:
            #is_exist[i // 8][i % 8] = True
    return is_exist


# Classifies each square and outputs the list in Forsyth-Edwards Notation (FEN)
def classify_cells_to_array_simul(model, img_filename_list):
    pred_list = []
    is_exist = np.zeros((8,8), dtype=bool)
    img_list = np.zeros((64,224,224,3),dtype=float)
    index = 0
    outs = []
    #print(np.shape(img_list))
    start=time.time()
    for filename in img_filename_list:
        img = prepare_image(filename)
        img *= 1./255
        #print(np.shape(img))
        img_list[index] = img[0]
        #outs.append(model(img,training=False))
        index += 1
        #print(np.shape(img_list))
    end=time.time()
    print("image_prepare_time:",end-start)
    #print(img_list)
    #print(np.shape(img_list))
    start=time.time()
    outs = model.predict(img_list,verbose=1,batch_size=16,workers=4,use_multiprocessing=False)
    end=time.time()
    #print("prediction_time:",end-start)
    #outs = model(img_list,training=False)
    for out in outs:
        top_pred = np.argmax(out)
        pred_list.append(top_pred)
    for i, val in enumerate(pred_list):
        if val == 1:
            is_exist[i // 8][i % 8] = True
    return is_exist

#model = load_model(os.path.dirname(__file__)+'/model_VGG16_weight_empty_log10.h5')
#print('loaded')

def predict(img):
    time.sleep(1.0)
    print('sleeping')
    #return 1
    print('load')
    model = load_model(os.path.dirname(__file__)+'/model_VGG16_weight_empty_log10.h5')
    print('loaded')
    pred = model.predict(img,verbose=1)
    print(np.argmax(pred))
    return np.argmax(pred)

def predict_with_model(img,model):
    #time.sleep(1.0)
    #return 1
    pred = model.predict(img,verbose=0)
    print(np.argmax(pred))
    return np.argmax(pred)

class model_iter:
    #def __init__(self):
    def __getitem__(self, index):
        return 



def classify_cells_to_array_multi(img_filename_list, executor,models):
    pred_list = []
    is_exist = np.zeros((8,8), dtype=bool)
    images = []
    arg_tuples=[]
    index = 0
    for filename in img_filename_list:
        img = prepare_image(filename)
        img *= 1./255
        images.append(img)
        predict(img)
        #arg_tuples.append((img,models[index]))
        index += 1
    models = model_iter()
    pred_list = []
    print('multi_process start')
    pool = multiprocessing.Pool(processes=4)
    pred_list = pool.map(predict,images)
    #with ProcessPoolExecutor() as e:
      #for pred in list(e.map(predict_with_model, images, models)):
        #pred_list.append(pred)
    #for pred in list(executor.map(predict, images)):
        #print('pred',pred)
        #pred_list.append(pred)
    #for pred in list(executor.map(predict_with_model, arg_tuples)):
        #print('pred',pred)
        #pred_list.append(pred)
    print(pred_list)
    for i, val in enumerate(pred_list):
        if val == 1:
            is_exist[i // 8][i % 8] = True
    
    return is_exist

def isClockwise(x1, y1, x2, y2, x3, y3):
  return (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2) > 10

def piece_filter_low_canny(bound, chessboardImage):
  areaOfCell = np.zeros((8,8), dtype=int)
  diffAreaOfCell = np.zeros((8,8), dtype=int)
  img = cv2.cvtColor(chessboardImage, cv2.COLOR_BGR2GRAY)
  dst = cv2.Canny(img, 20, 40)
  #dst = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
  cv2.imwrite('./drawDiff.jpg', dst)
  for i in range(8):
    for j in range(8):
      b0x = bound[i][j][1] * 0.9 + bound[i+1][j+1][1] * 0.1
      b0y = bound[i][j][0] * 0.9 + bound[i+1][j+1][0] * 0.1
      b1x = bound[i+1][j][1] * 0.9 + bound[i][j+1][1] * 0.1
      b1y = bound[i+1][j][0] * 0.9 + bound[i][j+1][0] * 0.1
      b2x = bound[i+1][j+1][1] * 0.9 + bound[i][j][1] * 0.1
      b2y = bound[i+1][j+1][0] * 0.9 + bound[i][j][0] * 0.1
      b3x = bound[i][j+1][1] * 0.9 + bound[i+1][j][1] * 0.1
      b3y = bound[i][j+1][0] * 0.9 + bound[i+1][j][0] * 0.1

      ymin = math.ceil(min(b0y,b1y,b2y,b3y))
      ymax = math.floor(max(b0y,b1y,b2y,b3y))
      xmin = math.ceil(min(b0x,b1x,b2x,b3x))
      xmax = math.floor(max(b0x,b1x,b2x,b3x))
      for x in range(xmin,xmax+1):
        for y in range(ymin,ymax+1):
          c1 = isClockwise(b0x,b0y,b1x,b1y,x,y)
          c2 = isClockwise(b1x,b1y,b2x,b2y,x,y)
          c3 = isClockwise(b2x,b2y,b3x,b3y,x,y)
          c4 = isClockwise(b3x,b3y,b0x,b0y,x,y)
          if c1 == c2 == c3 == c4:
            areaOfCell[i][j] += 1
            if dst[x][y] > 125:
              diffAreaOfCell[i][j] += 1
  
  ratio_threshold = 0.01
  isExist = np.zeros((8,8), dtype=int)
  print(diffAreaOfCell/areaOfCell)
  for i in range(8):
    for j in range(8):
      isExist[i][j] = -1
      if areaOfCell[i][j] * ratio_threshold >= diffAreaOfCell[i][j]:
        isExist[i][j] = 0
  return isExist

def piece_filter_high_canny(bound, chessboardImage):
  areaOfCell = np.zeros((8,8), dtype=int)
  diffAreaOfCell = np.zeros((8,8), dtype=int)
  img = cv2.cvtColor(chessboardImage, cv2.COLOR_BGR2GRAY)
  dst = cv2.Canny(img, 40, 80)
  #dst = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
  cv2.imwrite('./drawDiff.jpg', dst)
  for i in range(8):
    for j in range(8):
      b0x = bound[i][j][1] * 0.9 + bound[i+1][j+1][1] * 0.1
      b0y = bound[i][j][0] * 0.9 + bound[i+1][j+1][0] * 0.1
      b1x = bound[i+1][j][1] * 0.9 + bound[i][j+1][1] * 0.1
      b1y = bound[i+1][j][0] * 0.9 + bound[i][j+1][0] * 0.1
      b2x = bound[i+1][j+1][1] * 0.9 + bound[i][j][1] * 0.1
      b2y = bound[i+1][j+1][0] * 0.9 + bound[i][j][0] * 0.1
      b3x = bound[i][j+1][1] * 0.9 + bound[i+1][j][1] * 0.1
      b3y = bound[i][j+1][0] * 0.9 + bound[i+1][j][0] * 0.1

      ymin = math.ceil(min(b0y,b1y,b2y,b3y))
      ymax = math.floor(max(b0y,b1y,b2y,b3y))
      xmin = math.ceil(min(b0x,b1x,b2x,b3x))
      xmax = math.floor(max(b0x,b1x,b2x,b3x))
      for x in range(xmin,xmax+1):
        for y in range(ymin,ymax+1):
          c1 = isClockwise(b0x,b0y,b1x,b1y,x,y)
          c2 = isClockwise(b1x,b1y,b2x,b2y,x,y)
          c3 = isClockwise(b2x,b2y,b3x,b3y,x,y)
          c4 = isClockwise(b3x,b3y,b0x,b0y,x,y)
          if c1 == c2 == c3 == c4:
            areaOfCell[i][j] += 1
            if dst[x][y] > 125:
              diffAreaOfCell[i][j] += 1
  
  ratio_threshold = 0.1
  isExist = np.zeros((8,8), dtype=int)
  print(diffAreaOfCell/areaOfCell);
  for i in range(8):
    for j in range(8):
      isExist[i][j] = -1
      if areaOfCell[i][j] * ratio_threshold <= diffAreaOfCell[i][j]:
        isExist[i][j] = 1
  return isExist

# Classifies each square and outputs the list in Forsyth-Edwards Notation (FEN)
def classify_cells_to_array_with_filter(model, img_filename_list):
    pred_list = []
    is_exist = np.zeros((8,8), dtype=bool)
    for filename in img_filename_list:
        img = prepare_image(filename)
        img *= 1./255
        out = model.predict(img)
        top_pred = np.argmax(out)
        pred_list.append(top_pred)

    for i, val in enumerate(pred_list):
        if val == 1:
            is_exist[i // 8][i % 8] = True
    return is_exist

def classify_cells_pre_keras_to_array(model, img_filename_list): 
    img_width, img_height = 224, 224
    pred_list = []
    is_exist = np.zeros((8,8), dtype=bool)
    
    for image_name in img_filename_list:
            img_predict = []
            print(image_name)
            img = Image.open(image_name)
            img = img.convert("RGB")
            img = img.resize((img_width, img_height))
            img_array = np.array(img, dtype=float)
            img_predict.append(img_array)
            for i in range(len(img_predict)):
                img_predict[i] *= 1./255
            img_predict = np.array(img_predict, dtype=float)
            out = model.predict(img)
            top_pred = np.argmax(out)
            pred_list.append(top_pred)
    for i, val in enumerate(pred_list):
        if val == 1:
            is_exist[i // 8][i % 8] = True
    return is_exist

def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Read image and do lite image processing
def read_img_and_rescale(file, points):
	img = cv2.imread(str(file), 1)

	W = 1000
	height, width, depth = img.shape
	#imgScale = width 
	imgScale = W / width
	newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
	img = cv2.resize(img, (int(newX), int(newY)))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray_blur = cv2.blur(gray, (5, 5))

	points *= imgScale
	return img, gray_blur
'''
def piece_detect(_frame,executor=None,filepath=os.path.dirname(__file__)+'/corner.npy'):
#def piece_detect(_frame,filepath=os.path.dirname(__file__)+'/corner.npy'):
    print('piece_detect:')
    #model = load_model(os.path.dirname(__file__)+'/model_VGG16_weight_empty_log10.h5')
    #models = []
    #for i in range(64):
        #print(i)
        #models.append(load_model(os.path.dirname(__file__)+'/model_VGG16_weight_empty_log10.h5'))
    print('loaded model')
    #camera,dist=get_calibration_data()
    #frame = cv2.convertScaleAbs(_frame,alpha = 1.2,beta=+15)
    #rescale(frame)
    #frame = undistort(img=frame,DIM=(640, 480),K=np.array(camera),D=np.array(dist))
    print('preprocessed')
    #cv2.imwrite('frame.jpeg', frame)
    corners = np.load(filepath)
    #img, gray_blur = read_img_and_rescale('frame.jpeg', corners)
    #write_crop_images(img, corners, 0)
    #print('cropped')
    img_filename_list = grab_cell_files()
    img_filename_list.sort(key=natural_keys)
    start = time.time()
    #pre_filter = piece_filter_high_canny(corners, img)
    #print(pre_filter)
    #is_exist = classify_cells_to_array(model, img_filename_list)
    is_exist = classify_cells_to_array(models[13], img_filename_list)
    #print('simul')
    #is_exist = classify_cells_to_array_simul(model, img_filename_list)
    #is_exist = classify_cells_to_array_multi(img_filename_list,executor,models)
    end = time.time()
    print('time',end-start)

    print('classified')
    print(is_exist)
    return is_exist
'''
def piece_detect_test(_frame=None,executor=None,filepath=os.path.dirname(__file__)+'/corner.npy'):
#def piece_detect(_frame,filepath=os.path.dirname(__file__)+'/corner.npy'):
    executor = ProcessPoolExecutor(1)
    #executor = ThreadPoolExecutor(4)
    model = load_model(os.path.dirname(__file__)+'/model_VGG16_weight_empty_log10.h5')
    models = []
    #for i in range(64):
        #print(len(models))
        #models.append(load_model(os.path.dirname(__file__)+'/model_VGG16_weight_empty_log10.h5'))
    #models[3] = None
    while True:
        allstart = time.time()
        print('piece_detect:')
        #for i in range(64):
            #model[i] = load_model(os.path.dirname(__file__)+'/model_VGG16_weight_empty_log10.h5')
        print('loaded model')
        camera,dist=get_calibration_data()
        frame = cv2.convertScaleAbs(_frame,alpha = 1.2,beta=+15)
        rescale(frame)
        frame = undistort(img=frame,DIM=(640, 480),K=np.array(camera),D=np.array(dist))
        print('preprocessed')
        cv2.imwrite('frame2.jpeg', frame)
        corners = np.load(filepath)
        print(corners)
        img, gray_blur = read_img_and_rescale('frame2.jpeg', corners)
        write_crop_images(img, corners, 0)
        start=time.time()
        print('cropped')
        print('crop:',time.time()-start)
        img_filename_list = grab_cell_files()
        img_filename_list.sort(key=natural_keys)
        start = time.time()
        #pre_filter = piece_filter_high_canny(corners, img)
        #print(pre_filter)
        #is_exist = classify_cells_to_array(models[13], img_filename_list)
        #is_exist = classify_cells_to_array(models[3], img_filename_list)
        is_exist = classify_cells_to_array(model, img_filename_list)
        #is_exist = classify_cells_to_array_simul(model, img_filename_list)
        #is_exist = classify_cells_to_array_simul_multi(model, img_filename_list,executor)
        #is_exist = classify_cells_to_array_multi(img_filename_list,executor,models)
        end = time.time()
        print('time',end-start)

        print('classified')
        print(is_exist)
        print('alltime',allstart-time.time())
    return is_exist

def exist_to_fen(exist):
    fen = ""
    for i in range(8):
        empty_count = 0
        for j in range(8):
            if exist[7-j][i]:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += "P"
            else:
                empty_count += 1
        if empty_count > 0:
            fen += str(empty_count)
        if i != 7:
            fen += '/'
    return fen
# Converts the FEN into a PNG file
def fen_to_image(fen):
    print("fen:"+fen)
    board = chess.Board(fen)
    current_board = chess.svg.board(board=board)

    output_file = open('current_board.svg', "w")
    output_file.write(current_board)
    output_file.close()

    svg = svg2rlg('current_board.svg')
    renderPM.drawToFile(svg, 'current_board.png', fmt="PNG")
    return board
def undistort(img=None,DIM=(1920, 1080),K=np.array([[  1.08515378e+03,   0.00000000e+00,   9.83885166e+02],
       [  0.00000000e+00,   1.08781630e+03,   5.36422266e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]),D=np.array([[0.0], [0.0], [0.0], [0.0]])):
    # h,w = img.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)



DETECT_URL = 'http://127.0.0.1:8000/detect/detect/'

def piece_detect_with_server(frame):
    start=time.time()
    csv_mimetype = 'text/csv'
    jpeg_mimetype = 'image/jpeg'
    corners = np.load(os.path.dirname(__file__)+'/corner.npy')
    #corners = corners.astype('int32')
    #data = {"corner_byte":base64.b64encode(corners.tobytes())}
    data = {"array":corners.flatten().tolist()}
    cv2.imwrite('frame.jpeg', frame)
    img_name = 'frame.jpeg'
    img_data = open(img_name, 'rb').read()
    camera_name = os.path.dirname(__file__) + '/camera.csv'
    camera_data = open(camera_name, 'rb').read()
    dis_name = os.path.dirname(__file__) + '/dis.csv'
    dis_data = open(dis_name, 'rb').read()
    #print(corners)
    #print(corners.dtype)
    #print(np.shape(corners))
    #print(str(data['corner_byte']))
    #print(np.frombuffer(corners.tobytes()))
    files = {'img': (img_name, img_data, jpeg_mimetype), 
    'camera':(camera_name,camera_data,csv_mimetype),
    'dis':(dis_name,dis_data,csv_mimetype),
    }
    while True:
        try:
            response = requests.post(DETECT_URL, data=data, files=files, timeout=(3.0,6.0))
            break
        except requests.exceptions.Timeout:
            pass
    pred_list = response.json()['data']
    is_exist = np.zeros((8,8), dtype=bool)
    for i, val in enumerate(pred_list):
        if val == 1:
            is_exist[i // 8][i % 8] = True
    print(is_exist)
    print('Time',time.time()-start)
    return is_exist

TRAIN_URL = 'http://127.0.0.1:8000/detect/train/'

def piece_detect_with_train(frame, answer, is_save):
    start=time.time()
    csv_mimetype = 'text/csv'
    jpeg_mimetype = 'image/jpeg'
    corners = np.load(os.path.dirname(__file__)+'/corner.npy')
    #corners = corners.astype('int32')
    #data = {"corner_byte":base64.b64encode(corners.tobytes())}
    data = {"array":corners.flatten().tolist(), "answer":answer.flatten().astype(int).tolist(), "is_save": is_save}
    cv2.imwrite('frame.jpeg', frame)
    img_name = 'frame.jpeg'
    img_data = open(img_name, 'rb').read()
    camera_name = os.path.dirname(__file__) + '/camera.csv'
    camera_data = open(camera_name, 'rb').read()
    dis_name = os.path.dirname(__file__) + '/dis.csv'
    dis_data = open(dis_name, 'rb').read()
    files = {'img': (img_name, img_data, jpeg_mimetype), 
    'camera':(camera_name,camera_data,csv_mimetype),
    'dis':(dis_name,dis_data,csv_mimetype),
    }
    while True:
        try:
            response = requests.post(TRAIN_URL, data=data, files=files, timeout=(3.0,6.0))
            break
        except requests.exceptions.Timeout:
            pass
    pred_list = response.json()['data']
    is_exist = np.zeros((8,8), dtype=bool)
    for i, val in enumerate(pred_list):
        if val == 1:
            is_exist[i // 8][i % 8] = True
    print(is_exist)
    print('Time',time.time()-start)
    return is_exist
    
    



if __name__ == '__main__':
    #cap = cv2.VideoCapture(0)	
    #ret, frame = cap.read()
    frame = cv2.imread('frame.jpeg')
    #piece_detect_with_server(frame)
    answer = np.zeros((8,8),dtype=bool)
    for i in range(8):
        for j in range(8):
            answer[i][j] = ((i + j) % 2 == 1)
    piece_detect_with_train(frame,answer)
