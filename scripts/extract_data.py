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
TARGET_DISTANCE = 50.0

def getArgs():
    # input is the .tsv file generated from lane simulator when using --save_lane_result true
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str, default='', help='output absolute path')

    parser.add_argument('--bag_file', help="parse the lane_path from bag to evaluate lane meteics")

    parser.add_argument('--calib_file', type=str, help='calib_file path of front left camera')
    parser.add_argument('--target_distance', type=float, default=TARGET_DISTANCE, help='')

    args = parser.parse_args()

    return args

class CalibLoader:
    def __init__(self, calib_file):
        self.height = None
        self.width = None
        self.R = None
        self.P = None
        self.M = None
        self.D = None
        self.camera2imu = None

        self.map1 = None
        self.map2 = None

        self.load_calib_yaml_into_config(calib_file)

    def load_calib_yaml_into_config(self, calib_file):
        assert os.path.exists(calib_file), 'invalid camera calibration file : ' + str(calib_file)

        mono_calibration_fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        self.height = int(mono_calibration_fs.getNode('height').real())
        self.width = int(mono_calibration_fs.getNode('width').real())

        self.M = mono_calibration_fs.getNode('M').mat()

        d = mono_calibration_fs.getNode('D').mat()
        if d.shape[0] < d.shape[1]:
            self.D = d[0]
        else:
            self.D = d.transpose()[0]
        self.R = mono_calibration_fs.getNode('R').mat()
        self.P = mono_calibration_fs.getNode('P').mat()
        self.camera2imu = mono_calibration_fs.getNode('Tr_cam_to_imu').mat()

        self.imu2camera = np.linalg.inv(self.camera2imu)

        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.M,
                                                           self.D,
                                                           self.R,
                                                           self.P,
                                                           (self.width, self.height),
                                                           cv2.CV_16SC2)

    def print_calib_matrix_for_check(self):
        print('========== mono calibration mat: ==========')
        print( 'raw_camera_intrinsics : '      )
        print( self.M                          )
        print( 'distortion_coefficients : '    )
        print( self.D                          )
        print( 'R : '                          )
        print( self.R                          )
        print( 'P : '                          )
        print( self.P                          )
        print( 'height=', self.height          )
        print( 'width=', self.width            )
        print( 'camera2imu=\n', self.camera2imu)
        print( 'imu2camera=\n', self.imu2camera)

    def unwarp(self, img):
        undistorted_image = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        return undistorted_image

def transformPoint(pose, point):
    """ transform a point by the given pose
    :param pose: 4*4 matrix
    :param point: point is list type, which len is 3
    :return: the new point after transformation
    """
    if point is None or pose is None:
        return None

    point_homogeneous = copy.deepcopy(point)
    point_homogeneous.append(1)
    point_new = pose.dot(np.transpose([np.array(point_homogeneous)])).T[0]
    point_new = point_new[:-1] / point_new[-1]
    out = []
    out.append(point_new[0])
    out.append(point_new[1])
    out.append(point_new[2])
    return out

def parsePoseFromPose3dPb(pose3d):
    quaternion = (
        pose3d.orientation.x,
        pose3d.orientation.y,
        pose3d.orientation.z,
        pose3d.orientation.w)
    pose = transformations.quaternion_matrix(quaternion)
    pose[0, 3] = pose3d.position.x
    pose[1, 3] = pose3d.position.y
    pose[2, 3] = pose3d.position.z
    return pose

def find_nearest_index_sorted(start_idx, samples, ts):
    last_target_ts = 0.0
    target_idx = len(samples)
    for i in range(start_idx, len(samples)):
        if samples[i][0] >= ts:
            if abs(samples[i][0] - ts) < abs(last_target_ts - ts):
                target_idx = i
            else:
                target_idx = max(0, i-1)
            break
        else:
            last_target_ts = samples[i][0]
    return target_idx

def find_nearest_index_sorted_by_pose(start_idx, pose_samples, target_pose):
    target_idx = len(pose_samples)
    for i in range(start_idx, len(pose_samples)):
        pose = pose_samples[i][1]
        pose_global_2_local = np.linalg.inv(pose)
        target_pose_imu = transformPoint(pose_global_2_local, target_pose)
        if target_pose_imu[0] <= 0:
            target_idx = i
            break
    return target_idx


def assign_label(camera_samples, pose_samples, roll_angle_samples, lane_path_roll_samples):
    results = []
    last_pose_idx = 0
    last_roll_idx = 0
    last_target_ts_roll_idx = 0
    last_target_pose_idx = 0
    last_target_pose_roll_idx = 0
    last_lane_path_roll_idx = 0
    for camera_sample in camera_samples:
        sample = {'image_index': camera_sample[1], 'camera_path': camera_sample[2]}
        ts = camera_sample[0]
        target_idx = find_nearest_index_sorted(last_pose_idx, pose_samples, ts)
        last_pose_idx = target_idx
        sample['target_ts'] = ts + 5.0
        if target_idx < len(pose_samples):
            sample['pose_ts'] = pose_samples[target_idx][0]
            sample['pose'] = pose_samples[target_idx][1]
            pose_local_2_global = sample['pose']
            pose_global_2_local = np.linalg.inv(pose_local_2_global)
            # get pose position in imu coord
            imu_pose = transformPoint(pose_global_2_local, [pose_local_2_global[0][3], pose_local_2_global[1][3], pose_local_2_global[2][3]])
            # print 'debug imu pose: ', imu_pose
            target_imu_pose = [imu_pose[0] + args.target_distance, imu_pose[1], imu_pose[2]]
            target_global_pose = transformPoint(pose_local_2_global, target_imu_pose)

            target_idx = find_nearest_index_sorted_by_pose(last_target_pose_idx, pose_samples, target_global_pose)
            last_target_pose_idx = target_idx
            if target_idx < len(pose_samples):
                target_pose_ts = pose_samples[target_idx][0]
                sample['target_pose_ts' ] = target_pose_ts
                target_idx = find_nearest_index_sorted(last_target_pose_roll_idx, roll_angle_samples, target_pose_ts)
                last_target_pose_roll_idx = target_idx
                if target_idx < len(roll_angle_samples):
                    sample['target_pose_roll_ts'] = roll_angle_samples[target_idx][0]
                    sample['target_pose_roll_angle'] = roll_angle_samples[target_idx][1]
                    
        target_idx = find_nearest_index_sorted(last_roll_idx, roll_angle_samples, ts)
        last_roll_idx = target_idx
        if target_idx < len(roll_angle_samples):
            sample['curr_roll_ts'] = roll_angle_samples[target_idx][0]
            sample['curr_roll_angle'] = roll_angle_samples[target_idx][1]

        target_idx = find_nearest_index_sorted(last_target_ts_roll_idx, roll_angle_samples, sample['target_ts'])
        last_target_ts_roll_idx = target_idx
        if target_idx < len(roll_angle_samples):
            sample['target_ts_roll_ts'] = roll_angle_samples[target_idx][0]
            sample['target_ts_roll_angle'] = roll_angle_samples[target_idx][1]

        target_idx = find_nearest_index_sorted(last_lane_path_roll_idx, lane_path_roll_samples, ts)
        last_lane_path_roll_idx = target_idx
        if target_idx < len(lane_path_roll_samples):
            sample['lane_path_ts_roll_ts'] = lane_path_roll_samples[target_idx][0]
            sample['lane_path_ts_roll_angle'] = lane_path_roll_samples[target_idx][1]

        if 'target_ts_roll_angle' in sample and 'curr_roll_angle' in sample and 'target_pose_roll_angle' in sample:
            results.append([camera_sample[0], sample])

    with open(os.path.join(args.output_path, 'result.tsv'), 'w') as result_file:
        for sample in results:
            result_file.write("%s\t%d\t%f\t%f\t%f\t%f\n" % (str(format(sample[0], '.2f')), sample[1]['image_index'], sample[1]['curr_roll_angle'], sample[1]['target_ts_roll_angle'], sample[1]['target_pose_roll_angle'], sample[1]['lane_path_ts_roll_angle']))

    return results

def visualize_samples(output_folder, samples):
    for sample_elem in samples:
        ts = sample_elem[0]
        sample = sample_elem[1]
        raw_image = cv2.imread(sample['camera_path'])
        image_index = sample['image_index']
        cv2.putText(raw_image, 'imu roll: %.2f' % sample['curr_roll_angle'], (10, 26), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        cv2.putText(raw_image, 'target ts roll: %.2f' % sample['target_ts_roll_angle'], (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        cv2.putText(raw_image, 'target pose roll: %.2f' % sample['target_pose_roll_angle'], (10, 74), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        cv2.putText(raw_image, 'lane path ts roll: %.2f' % sample['lane_path_ts_roll_angle'], (10, 96), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        output_path = os.path.join(output_folder + '/vis', ('%.2f' % ts) + '_' + str('%05d' % image_index) + '.jpg')
        cv2.imwrite(output_path, raw_image)
        if abs(float(sample['curr_roll_angle']) - float(sample['target_pose_roll_angle'])) > 0.5:
            cv2.imwrite(os.path.join(output_folder + '/important_vis', ('%.2f' % ts) + '_' + str('%05d' % image_index) + '.jpg'), raw_image)

def main():
    global args
    args = getArgs()
    bag_file = args.bag_file
    if '.bag' in bag_file:
        bag = rosbag.Bag(bag_file)
    elif '.db' in bag_file:
        bag = fastbag.Reader(bag_file)
        bag.open()
    else:
        print("failed parsing bag: ", bag_file)
        exit()

    print 'Target distance=', args.target_distance
    
    last_odom_sec = 0.0
    last_camera_sec = 0.0
    calib = CalibLoader(args.calib_file)
    bridge = CvBridge()
    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        if not os.path.exists(args.output_path + '/raw'):
            os.makedirs(args.output_path + '/raw')
        if not os.path.exists(args.output_path + '/vis'):
            os.makedirs(args.output_path + '/vis')
        if not os.path.exists(args.output_path + '/important_vis'):
            os.makedirs(args.output_path + '/important_vis')
            
    image_height = None
    image_width = None
    camera_samples = []
    roll_angle_samples = []
    lane_path_roll_samples = []
    pose_samples = []
    image_index = 0
    for topic, msg, t in bag.read_messages(topics=['/planning/trajectory', '/perception/lane_path', '/navsat/odom', '/front_left_camera/image_color/compressed', '/vehicle/control_cmd', '/perception/lane_path']):
        if topic == '/navsat/odom':
            curr_sec = float(str(msg.header.stamp)) / 1000000000.0
            # print curr_sec
            #interval = curr_sec - last_odom_sec
            #print ("%f" % interval)
            #last_odom_sec = curr_sec
            # print msg.pose.pose
            pose = parsePoseFromPose3dPb(msg.pose.pose)
            pose_samples.append([curr_sec, pose])
        elif topic == '/front_left_camera/image_color/compressed':
            curr_sec = float(str(msg.header.stamp)) / 1000000000.0
            time_str = str(format(curr_sec, '.2f'))
            # print curr_sec
            #interval = curr_sec - last_camera_sec
            #print ("%f" % interval)
            last_camera_sec = curr_sec
            undistorted_image = calib.unwarp(bridge.compressed_imgmsg_to_cv2(msg))
            if image_height is None:
                image_height = undistorted_image.shape[0]
            if image_width is None:
                image_width = undistorted_image.shape[1]
            if args.output_path is not None:
                file_path = os.path.join(args.output_path + '/raw', time_str + '_' + str('%05d' % image_index) + '.jpg')
                cv2.imwrite(file_path, undistorted_image)
            camera_samples.append([curr_sec, image_index, file_path])
            image_index += 1

        elif topic == '/vehicle/control_cmd':
            control_cmd = ControlCommand()
            control_cmd.ParseFromString(msg.data)
            # print control_cmd.header.timestamp_msec / 1000.0, control_cmd.debug_cmd.roll_angle
            roll_angle_samples.append([control_cmd.header.timestamp_msec / 1000.0, control_cmd.debug_cmd.roll_angle])

        elif topic == '/perception/lane_path':
            lane_detection = LaneDetection()
            lane_detection.ParseFromString(msg.data)
            lane_path_roll_samples.append([lane_detection.header.timestamp_msec / 1000.0, lane_detection.road_semantics.road_roll_degree])

    print 'finish reading msg, start labeling'
    camera_samples.sort(key=lambda x: x[0])
    pose_samples.sort(key=lambda x: x[0])
    roll_angle_samples.sort(key=lambda x: x[0])
    lane_path_roll_samples.sort(key=lambda x: x[0])

    labeled_samples = assign_label(camera_samples, pose_samples, roll_angle_samples, lane_path_roll_samples)
    print 'start generate visualize images'
    visualize_samples(args.output_path, labeled_samples)
    
if __name__ == '__main__':
    main()
