#!/usr/bin/env python

import sys
import glob
import serial
from serial import Serial
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import drawnow
import cv2
import csv
import threading  #http://pythonstudy.xyz/python/article/24-%EC%93%B0%EB%A0%88%EB%93%9C-Thread

# Gps Module device connection port
# ser = Serial('/dev/ttyUSB0', 115200)
# openStreetMap https://www.openstreetmap.org/
# Export manually
points = (37.2180, 127.0461, 37.1565, 127.1359)

def serial_ports():
    ser = Serial('/dev/ttyUSB0', 115200)
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return print(result)


def serial_test_gps():
    for i in range(10):
        if ser.readable():
            res = ser.readline()
            print(res)
            msg = res.decode('utf-8')[:len(res)-1]
            latitude = msg[10:18]
            longitude = msg[20:29]
            print(i, "Latitude :", latitude, "longitude:", longitude)


def gps_write_test():
    ## file write test ##
    # .csv  https://devpouch.tistory.com/55
    f = open('gps-test.csv','w', newline='')
    # f = open("test.txt", 'w')
    wr = csv.writer(f)

    for i in range(10):
        if ser.readable():
            res = ser.readline()
            msg = res.decode('utf-8')[:len(res)-1]
            latitude = msg[10:18]
            longitude = msg[20:29]
            print(i, "Latitude :", latitude, "longitude:", longitude)
        wr.writerow([latitude, longitude])
        # f.write(latitude+"  "+longitude+"\n")

    f.close()
    ## file read test (text)
    # f = open("test.txt", 'r')
    # data = f.read()
    # print(data)
    ## csv
    f = open('gps-test.csv','r')
    rdr = csv.reader(f)
    for line in rdr:
        print(line)

    f.close()


def get_gps():
    # GPS data save (Neo 6m GPS Module)
    minute = 1
    saving_time = minute*60
    init = time.time()
    f = open('gps-test.csv','w', newline='')
    wr = csv.writer(f)
    while True:
        if ser.readable():
            res = ser.readline()
            msg = res.decode()[:len(res)-1]
            latitude = msg[10:18]
            longitude = msg[20:29]
        wr.writerow([latitude, longitude])
        time_ = time.time()-init
        if time_ > saving_time:
            break
    f.close()


def scale_to_img(lat_lon, h_w):

    old = (points[2], points[0])
    new = (0, h_w[1])
    y = ((lat_lon[0] - old[0]) * (new[1] - new[0]) /
         (old[1] - old[0])) + new[0]
    old = (points[1], points[3])
    new = (0, h_w[0])
    x = ((lat_lon[1] - old[0]) * (new[1] - new[0]) /
         (old[1] - old[0])) + new[0]
    # y must be reversed because the orientation of the image in the matplotlib.
    # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
    return int(x), h_w[1] - int(y)


def create_img(color, width=2):
    data = pd.read_csv('gps.csv', names=['LATITUDE', 'LONGITUDE'], sep=',')

    result_image = Image.open('map_new.png', 'r')
    img_points = []
    gps_data = tuple(zip(data['LATITUDE'].values, data['LONGITUDE'].values))
    print(len(gps_data))
    for d in gps_data[:50]:
        x1, y1 = scale_to_img(d, (result_image.size[0], result_image.size[1]))
        img_points.append((x1, y1))
        draw = ImageDraw.Draw(result_image)
        draw.line(img_points, fill=color, width=width)
    #drawing line
    draw = ImageDraw.Draw(result_image)
    draw.line(img_points, fill=color, width=width)

    return result_image


def get_ticks():
    x_ticks = []
    y_ticks = []
    x_ticks = map(lambda x: round(x, 4), np.linspace(
        points[1], points[3], num=7))
    y_ticks = map(lambda x: round(x, 4), np.linspace(
        points[2], points[0], num=8))
    y_ticks = sorted(y_ticks, reverse=True)

    return x_ticks, y_ticks


def read_gps(filename,int_time):
    data = pd.read_csv(filename, names=['LATITUDE', 'LONGITUDE'], sep=',')
    result_image = Image.open('map_new.png', 'r')
    img_points = []
    gps_data = tuple(zip(data['LATITUDE'].values, data['LONGITUDE'].values))
    for d in gps_data:
        x1, y1 = scale_to_img(d, (result_image.size[0], result_image.size[1]))
        img_points.append((x1, y1))
        draw = ImageDraw.Draw(result_image)
        draw.line(img_points, (0, 0, 255), width=3)

        cv2_img = np.array(result_image)
        # img = cv2.imread('map_1.png', cv2.IMREAD_COLOR)
        img1 = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.putText(img1, "Latitude : %0.5f" % d[0], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img1, "Longitude: %0.5f" % d[1], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(img1, (x1,y1), 3, (0, 0, 255), cv2.FILLED, cv2.LINE_4)
        cv2.namedWindow('Navigation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Navigation', 800, 600)
        cv2.imshow("Navigation", img1)
        if cv2.waitKey(1) == 27:
            break
        time.sleep(int_time)

    # cv2.destroyAllWindows()

def video_cap ():

    cap=cv2.VideoCapture(0)

    if not cap.isOpened():
        print("camera open failed!")
        sys.exit()

    w=round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame=cap.read()
        if not ret:
            break

        # frame_resize=cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('video', frame)

        if cv2.waitKey()== 27:
            cap.release()
            break

def main(test):
    result_image1 = Image.open('map_new.png', 'r')
    img_points1 = []
    c = 0
    b = 0

    # test = True
    latitude = 37.163
    longitude = 127.0673

    time_ = time.time()
    f = open('gps_'+str(time_)+'.csv','w', newline='')
    # f = open("test.txt", 'w')
    wr = csv.writer(f)

    while True:
        
        if ser.readable():
            res = ser.readline()
            msg = res.decode()[:len(res)-1]

            if test:
                latitude += 0.0005
                longitude += 0.0005
            else:
                latitude = msg[10:18]
                longitude = msg[20:29]

            wr.writerow([latitude, longitude])

        try:
            lat = float(latitude)
            logt = float(longitude)
            d = (lat, logt)
            # print(d)
            x1, y1 = scale_to_img(
                d, (result_image1.size[0], result_image1.size[1]))
            # print(x1, y1)
            img_points1.append((x1, y1))
            draw = ImageDraw.Draw(result_image1)
            draw.line(img_points1, (0, 0, 255), width=3)
            cv2_img = np.array(result_image1)
            # cv2.putText(img1, "Latitude : %0.5f" % latitude, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(img1, "Longitude: %0.5f" % longitude, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.circle(img1, (x1,y1), 3, (0, 0, 255), cv2.FILLED, cv2.LINE_4)
        except:
            b += 1

        cv2_img1 = np.array(result_image1)
        img1 = cv2.cvtColor(cv2_img1, cv2.COLOR_RGB2BGR)
        
        # cv2.putText(img1, "Latitude : %0.5f" % latitude, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(img1, "Longitude: %0.5f" % longitude, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.circle(img1, (x1,y1), 3, (0, 0, 255), cv2.FILLED, cv2.LINE_4)
        cv2.namedWindow('Navigation', cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Navigation", img1)

        if cv2.waitKey(1) == 27:
            f.close()
            break

        # time.sleep(0.01)
        # c += 1
        # print(c)
        # if c > 50:
        #     break
    

if __name__ == '__main__':

    # serial_ports()
    # gps_write_test()
    # get_gps()

    # # 데몬 쓰레드
    # t1 = threading.Thread(target=video_cap)
    # t1.daemon = True 
    # t1.start()
    read_gps('gps_test2.csv',0.12)
    # main(False)
    cv2.waitKey()
    cv2.destroyAllWindows()
    sys.exit()
