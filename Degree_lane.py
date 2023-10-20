#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import random
import time
import numpy as np
import math
import cv2
from alexnet import alexnet
import tensorflow as tf
from ultralytics import YOLO
from keras.applications.mobilenet_v2 import preprocess_input

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

IM_WIDTH= 455
IM_HEIGHT=256

LR = 1e-3
EPOCHS = 4
# MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)
# t_time = 0.2
# model = alexnet(IM_WIDTH, IM_HEIGHT, LR)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model= tf.keras.models.load_model(MODEL_NAME)

# model.trainable=False

def nothing(x):
    pass

# cv2.namedWindow("Trackbars")

# cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
# cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# last_time = time.time()
# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)

def predict(image):
    print("SPAWN POINT", sp)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IM_WIDTH,IM_HEIGHT))

    # predict=model.predict([image.reshape(1,IM_HEIGHT,IM_WIDTH,3)])[0]
    # print("#############################",predict)
    # bl = (0,250)
    # tl = (120 ,130)
    # br = (455,250)
    # tr = (330,130)
    # bl = (0,250)
    # tl = (110 ,130)
    # br = (455,250)
    # tr = (330,130)
    bl = (0,250)
    tl = (122 ,120)
    br = (455,250)
    tr = (322,120)

    cv2.circle(image, tl, 5, (0,0,255), -1)
    cv2.circle(image, bl, 5, (0,0,255), -1)
    cv2.circle(image, tr, 5, (0,0,255), -1)
    cv2.circle(image, br, 5, (0,0,255), -1)

    cv2.imshow("image",image)

    ## Aplying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
    
    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(image, matrix, (640,480))

    cv2.imshow("perpective",transformed_frame)

    

    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2HSV)

    # l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    # l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    # l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    # u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    # u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    # u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        
    # lower = np.array([l_h,l_s,l_v])
    # upper = np.array([u_h,u_s,u_v])
    # lower = np.array([30,5,130])
    # upper = np.array([255,50,255])
    # lower = np.array([30,0,130])
    # upper = np.array([255,50,255])
    # lower = np.array([50,4,140])
    # upper = np.array([255,50,255])
    # lower = np.array([0,4,130])
    # upper = np.array([255,50,255])
    lower = np.array([5,5,140])
    upper = np.array([255,50,255])
    # lower_y = np.array([50,50,50])
    # upper_y = np.array([110,255,255])
    # lower_w = np.array([210,210,210])
    # upper_w = np.array([255,255,255])

    mask = cv2.inRange(hsv_transformed_frame, lower, upper)
    noiseless_image_bw = cv2.medianBlur(mask, 7)
    kernel = np.ones((5, 5), np.uint8)
    # noiseless = cv2.erode(mask,kernel)
    # noiseless = cv2.dilate(noiseless,kernel)
    # mask_y = cv2.inRange(hsv_transformed_frame, lower_y, upper_y)
    # mask_w = cv2.inRange(hsv_transformed_frame, lower_w, upper_w)
    # cv2.imshow("mask_y",mask_y)
    # cv2.imshow("mask_w",mask_w)
    cv2.imshow("denoised mask", noiseless_image_bw)
    noiseless_image_bw = cv2.dilate(noiseless_image_bw,kernel)
    # cv2.imshow("ineki", noiseless_just_looking)

    mask=noiseless_image_bw
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    # print("midpoint",midpoint,mask.shape[0])
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
        

    #Sliding Window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()
    middles=[]
    middles_height=[]
    cv2.imshow("mask",mask)
    while y>0:
        ## Left threshold
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50 + cx)
                left_base = left_base-50 + cx
            
        ## Right threshold
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(right_base-50 + cx)
                right_base = right_base-50 + cx
        
        cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-40), (255,255,255), 2)
        # print("rectangles",left_base,right_base,y)
        y -= 40
        middles.append(left_base+((right_base-left_base)/2))
        middles_height.append(y)

    # print("middles",middles[4],middles[0])
    # print("height",middles_height[4],middles_height[0])
    angle = math.degrees(math.atan2((middles_height[0] - middles_height[4]),(middles[4] - middles[0])))
    print("anglesss",angle)

    cv2.imshow("Lane Detection - Sliding Windows", msk)
    print("middles of the picture",(middles[4] - middles[0]),(middles[4] - middles[0])-455/2)
    if(angle==90):
        vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
    # if(angle > 90 and angle < 140):
    #     print("@@@@@@@@@@@@@@@@@@@@@@@@@left")
    #     vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-0.1))
    # if(angle > 140 and angle < 170):
    #     vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-0.2))
    # if(angle < 90 and angle >40):
    #     print("@@@@@@@@@@@@@@@@@@@@@@@@@right")
    #     vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.1))
    # if(angle < 40):
    #     print("@@@@@@@@@@@@@@@@@@@@@@@@@right")
    #     vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.2))
    if((0 < angle) and (angle <180) and (angle !=90)):
        vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-1*(angle-90)/(90)))
        print("applied",(angle-90)/90)
    


    cv2.waitKey(1)



def process_img(image):
    i=np.array(image.raw_data)
    i2=i.reshape((IM_HEIGHT,IM_WIDTH, 4))
    i3=i2[:, :, :3]
    i3 = cv2.cvtColor(i3, cv2.COLOR_BGR2RGB)

    model = YOLO("yolov8n.pt")
    result = model(i3)
    # result = model.track(source=i3 , show=True, tracker="bytetrack.yaml")

    result.print()
    # result.show()
    # print("YOLO results", result)
    # cv2.imshow("YOLO", result)

    cv2.imshow("hi",i3)
    cv2.waitKey(1)
    preprocess_input(i3)
    predict(i3)

    return i3

actor_list=[]


try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    # spawn_point = random.choice(world.get_map().get_spawn_points())
    spawn_point = carla.Transform(carla.Location(x=73.1934 , y=-136.704, z=9.8374), carla.Rotation(pitch=-0.647311, yaw=-178.773, roll=8.33817e-10))
    # actor = world.spawn_actor(blueprint, transform)
    vehicle = world.spawn_actor(bp, spawn_point)
    sp=spawn_point
    print("SPAWN POINT", spawn_point)
    # vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    # spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    spawn_point = carla.Transform(carla.Location(x=2.5, z=1.4),carla.Rotation(-15.0,0,0))


    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))

    # do something with this sensor
    sensor.listen(lambda data: process_img(data))

    while(True):
        time.sleep(600)
        break

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')


