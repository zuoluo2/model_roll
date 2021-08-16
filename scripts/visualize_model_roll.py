#!/usr/bin/python

import numpy as np
from numpy import linalg as la
import datetime
import logging
import rospy
import rosbag
import gpxpy
import gpxpy.gpx
import os
import sys
import tf
# import pyproj_utm
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
import math
import tf.transformations as transform
import fastbag
import argparse
from collections import OrderedDict
import cv2
from cv_bridge import CvBridge
import tf.transformations as transformations
from perception.lane_detection_pb2 import LaneDetection
from planning.planning_trajectory_pb2 import PlanningTrajectory
from control.control_command_pb2 import ControlCommand
import copy

args = None

def getArgs():
    # input is the .tsv file generated from lane simulator when using --save_lane_result true
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str, default='', help='output absolute path')

    parser.add_argument('--model_result', help="parse the lane_path from bag to evaluate lane meteics")

    parser.add_argument('--label_file', type=str, help='calib_file path of front left camera')

    args = parser.parse_args()

    return args

        
def main():
    global args
    args = getArgs()

    samples = []
    with open(args.label_file, 'r') as label_file:
        for line in label_file:
            fs = line.rstrip('\r').split('\t')
            samples.append([fs[0], fs[1], fs[2]])
    if not os.path.exists(args.output_path + '/model_vis'):
        os.makedirs(args.output_path + '/model_vis')
    
    with open(args.model_result, 'r') as model_file:
        i = 0
        smooth_roll = None
        for line in model_file:
            fs = line.rstrip('\t').split('\t')
            image_path = fs[0]
            curr_pose_roll = fs[1]
            target_roll = fs[2]
            model_roll = fs[3]
            if len(fs) > 4:
                smooth_roll = fs[4]
            
            sample = samples[i]
            output_image_path = os.path.join(args.output_path, 'model_vis', samples[i][0] + ('_%05d' % int(samples[i][1])) + '.jpg')
            index = sample[1]
            raw_image = cv2.imread(image_path)
            cv2.putText(raw_image, 'frame: %d' % int(index), (10, 26), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
            cv2.putText(raw_image, 'imu roll: %.2f' % float(sample[2]), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
            cv2.putText(raw_image, 'target pose roll: %.2f' % float(target_roll), (10, 74), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
            cv2.putText(raw_image, 'orig model roll: %.2f' % float(model_roll), (10, 98), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
            if smooth_roll is not None:
                cv2.putText(raw_image, 'smoothed model roll: %.2f' % float(smooth_roll), (10, 122), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
            
            cv2.imwrite(output_image_path, raw_image)
            if i >= 2000:
                break
            i += 1
    
if __name__ == '__main__':
    main()
