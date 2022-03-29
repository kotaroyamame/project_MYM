import cv2
from calibration import get_calibration_data
from cv_chess_functions import read_img, undistort, points_by_points_with_twopoint_perspective, canny_edge
import numpy as np
import os
pt = []
def mouse_callback(event, x, y, flags, param):
    global pt

    #マウスの左ボタンがクリックされたとき
    if event == cv2.EVENT_LBUTTONDOWN and len(pt)<6:
        print('click!')
        print(x, y)
        print(pt)
        pt.append((x, y))

    #マウスの右ボタンがクリックされたとき
    if event == cv2.EVENT_RBUTTONDOWN:
        print(pt)
        pt.pop()
#画像左上:0, 画像右上:1, 画像左下:2, 画像右下:3
def prepare_corners_manual(a1, a8, filepath=os.path.dirname(__file__)+'/corner'):
    cap = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = cap.read()
        cv2.namedWindow('live', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('live', mouse_callback)
        camera,dist=get_calibration_data()
        frame = cv2.convertScaleAbs(frame,alpha = 1.1,beta=-30)
        #魚眼補正
        frame = undistort(img=frame,DIM=(640, 480),K=np.array(camera),D=np.array(dist))
        ptl = np.array( pt )
        frame_view = frame.copy()
        cv2.polylines(frame_view, [ptl] ,False,(200,10,10))
        # Display the resulting frame
        cv2.imshow('live', frame_view)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if len(pt)!=6:
                print('Plese set line')
                continue
            print('Working...')
            # Save the frame to be analyzed
            cv2.imwrite('frame.jpeg', frame)

            # Low-level CV techniques (grayscale & blur)
            img, gray_blur = read_img('frame.jpeg')
            frame2 = frame.copy()
            points = points_by_points_with_twopoint_perspective(pt)
            for _point in points:
                for __point in _point:
                    point = (int(__point[0]),int(__point[1]))
                    cv2.drawMarker(frame2, position=point, color=(0, 0, 255))
            cv2.imwrite('frame2.jpeg', frame2)
            if a1 % 2 == a8 % 2:
                points = np.transpose(points, (1,0,2))

            if a1 == 1 or a1 == 2:
                a1 = 3 - a1
            if a8 == 1 or a8 == 2:
                a8 = 3 - a8
            
            if a1 >= 2 and a8 >= 2:
                points = np.flipud(points)
                a1 -= 2
                a8 -= 2

            if a1 == 1 and a8 == 0:
                points = np.fliplr(points)
                a1 -= 1
                a8 += 1
            print(points)
            np.save(filepath, points)
            print('prepared!')
            cap.release()
            return True


if __name__ == '__main__':
  prepare_corners_manual(0, 1)