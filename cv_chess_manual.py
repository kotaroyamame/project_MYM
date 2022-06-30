import re
import cv2
from keras.models import load_model
import sys
import time
sys.path.append("./")
sys.path.append("./Data")
from cv_chess_functions import (classify_cells_pre_keras, points_by_points, points_by_points_with_perspective_projection, points_by_points_with_twopoint_perspective, read_img,
                               canny_edge,
                               hough_line,
                               h_v_lines,
                               line_intersections,
                               cluster_points,
                               augment_points,
                               write_crop_images,
                               grab_cell_files,
                               classify_cells,
                               fen_to_image,
                               piece_detect,
                               exist_to_fen,
                               atoi,undistort)
import numpy as np
from calibration import get_calibration_data
from concurrent.futures import ProcessPoolExecutor

# Resize the frame by scale by dimensions
def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


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




def main():
  print('read h5')
  # Load in the CNN model
  model = load_model('model_VGG16_weight_empty_log8.h5')
  print('end h5')
  # Select the live video stream source (0-webcam & 1-GoPro)
  cap = cv2.VideoCapture(0)
  ret, _frame = cap.read()
  time.sleep(1)
  ret, _frame = cap.read()
  # Show the starting board either as blank or with the initial setup
  # start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
  blank = '8/8/8/8/8/8/8/8'
  board = fen_to_image(blank)
  board_image = cv2.imread('current_board.png')
  cv2.imshow('img', board_image)
  print('endfen')
  #executor = ProcessPoolExecutor(10)
  while(True):
      # Capture frame-by-frame
      ret, _frame = cap.read()
      #exist = piece_detect(_frame,executor)
      exist = piece_detect(_frame)
      fen = exist_to_fen(exist)
      print(fen)
      board = fen_to_image(fen)
      frame = cv2.imread('frame.jpeg')
      cv2.imshow('live', frame)
      board_image = cv2.imread('current_board.png')
      cv2.imshow('img', board_image)
      print('Completed!')
      cv2.waitKey(3000)
  # When everything is done, release the capture
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()