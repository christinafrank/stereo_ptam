# Import of the libraries
import numpy as np

import time
from itertools import chain
from collections import defaultdict

# Import the other classes of the project 
from covisibility import CovisibilityGraph
from optimization import BundleAdjustment
from mapping import Mapping
from mapping import MappingThread
from components import Measurement
from motion import MotionModel
from loopclosing import LoopClosing

# THREAD - TRACKING
# For detecting the robots pose and creating a visual keyframe 
class Tracking(object):
    def __init__(self, params):
        
# Set the variables:
# Minimize the reprojection error through Bundle Adjustment
# Set minimum number of measured points
# Set maximum number of loop interations for map correction
        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

# STEP - REFINE POSE
    def refine_pose(self, pose, cam, measurements):
# Check if the minimum number of measured points is reached
        assert len(measurements) >= self.min_measurements, (
            'Not enough points')

# Clear Bundle Adjustment (reprojection error) and add the currently measured position of the robot            
        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, cam, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m)

        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)


# Main part of the algorithm SPTAM (Stereo Parallel Tracking and Mapping)
class SPTAM(object):
# Initialize the variables
# Set connection to the three threads: Tracking, Local Mapping, Loop Closure
# Tracking:         Detect the robots position and create a keyframe of comparison between 3D points and 2D map
# Local Mapping:    Refine 3D and 2D comparison, minimize the reprojection error and remove bad points
# Loop Closure:     Detect revisited places, estimate the relative transformation and correct the keyframes and map features
    def __init__(self, params):
        self.params = params
# Set thread Tracking
        self.tracker = Tracking(params)
        self.motion_model = MotionModel()
        
# Set thread Local Mapping
        self.graph = CovisibilityGraph()
        self.mapping = MappingThread(self.graph, params)
        
# Set thread Loop Closure
        self.loop_closing = LoopClosing(self, params)
        self.loop_correction = None
        
# Set the keyframe
        self.reference = None        # reference keyframe
        self.preceding = None        # last keyframe
        self.current = None          # current frame
        self.status = defaultdict(bool)
        
# Stop Loop Closure, if already executed       
    def stop(self):
        self.mapping.stop()
        if self.loop_closing is not None:
            self.loop_closing.stop()


    def initialize(self, frame):
# Create initial map
        mappoints, measurements = frame.triangulate()
        assert len(mappoints) >= self.params.init_min_points, (
            'Not enough points to initialize map.')

# Create initial keyframe from initial map
        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.graph.add_keyframe(keyframe)
        self.mapping.add_measurements(keyframe, mappoints, measurements)
        if self.loop_closing is not None:
            self.loop_closing.add_keyframe(keyframe)

# Set reference, preceding and current keyframe to initial keyframe
        self.reference = keyframe
        self.preceding = keyframe
        self.current = keyframe
        self.status['initialized'] = True

# Set initial pose of the robot
        self.motion_model.update_pose(
            frame.timestamp, frame.position, frame.orientation)

# THREAD - TRACKING
    def track(self, frame):
# While robot is not moving, wait
        while self.is_paused():
            time.sleep(1e-4)
# When robot is moving, start tracking
        self.set_tracking(True)

# STEP - FEATURE EXTRACTION: Capture the actual frame of the 3D world
        self.current = frame
        print('Tracking:', frame.idx, ' <- ', self.reference.id, self.reference.idx)

# STEP - POSE PREDICTION: Predict the current position of the robot
        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)

# While step Loop Closing, correct current pose
        if self.loop_closing is not None:
            if self.loop_correction is not None:
                
# Use g2o for pose graph optimization
                estimated_pose = g2o.Isometry3d(
                    frame.orientation,
                    frame.position)

# Create copy of the frame and execute correction
                estimated_pose = estimated_pose * self.loop_correction
                frame.update_pose(estimated_pose)
                self.motion_model.apply_correction(self.loop_correction)
                self.loop_correction = None

# STEP - MATCHING: Project map points and search for matches in the neighbourhood
        local_mappoints = self.filter_points(frame)
        measurements = frame.match_mappoints(
            local_mappoints, Measurement.Source.TRACKING)

        print('measurements:', len(measurements), '   ', len(local_mappoints))

# Use BRISK descriptor to describe the features of the points
# Compare the descriptors between map point and features
        tracked_map = set()
        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)

# STEP - POSE REFINEMENT: first get actual pose
        try:
            self.reference = self.graph.get_reference_frame(tracked_map)
# Compare the previous camera pose with the relative motion in the current, local frame
            pose = self.tracker.refine_pose(frame.pose, frame.cam, measurements)
# Update the pose
            frame.update_pose(pose)
            self.motion_model.update_pose(
                frame.timestamp, pose.position(), pose.orientation())
            tracking_is_ok = True
        except:
            tracking_is_ok = False
            print('tracking failed!!!')

# STEP - KEYFRAME SELECTION
        if tracking_is_ok and self.should_be_keyframe(frame, measurements):
            print('new keyframe', frame.idx)
            keyframe = frame.to_keyframe()
            keyframe.update_reference(self.reference)
            keyframe.update_preceding(self.preceding)

# Set new keyframe
            self.mapping.add_keyframe(keyframe, measurements)
# THREAD - LOOP CLOSURE
# Add the keyframe to see if this place already has been visited
            if self.loop_closing is not None:
                self.loop_closing.add_keyframe(keyframe)
            self.preceding = keyframe

        self.set_tracking(False)

# Helping method for STEP - MATCHING
    def filter_points(self, frame):
# Project the mappoints
        local_mappoints = self.graph.get_local_map_v2(
            [self.preceding, self.reference])[0]
# Set current view points
        can_view = frame.can_view(local_mappoints)
        print('filter points:', len(local_mappoints), can_view.sum(), 
            len(self.preceding.mappoints()),
            len(self.reference.mappoints()))
        
        checked = set()
        filtered = []
# Check if points are in current frame, if yes increase count
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

# Get all references for this points
        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

# Return filtered map points that are in the current view
        return filtered

# Helping method for STEP - KEYFRAME SELECTION
    def should_be_keyframe(self, frame, measurements):
        if self.adding_keyframes_stopped():
            return False

# Set actual matches and the ones of the current keyframe
        n_matches = len(measurements)
        n_matches_ref = len(self.reference.measurements())

        print('keyframe check:', n_matches, '   ', n_matches_ref)

# Set only as new keyframe, if minimum number of tracked points ratio is fullfilled
# or if the current keyframe has less than 20 matches
        return ((n_matches / n_matches_ref) < 
            self.params.min_tracked_points_ratio) or n_matches < 20


# THREAD - LOOP CLOSURE
# STEP - LOOP CORRECTION
    def set_loop_correction(self, T):
        self.loop_correction = T

# Other helping methods
    def is_initialized(self):
        return self.status['initialized']

    def pause(self):
        self.status['paused'] = True

    def unpause(self):
        self.status['paused'] = False

    def is_paused(self):
        return self.status['paused']

    def is_tracking(self):
        return self.status['tracking']

    def set_tracking(self, status):
        self.status['tracking'] = status

    def stop_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = True

    def resume_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = False

    def adding_keyframes_stopped(self):
        return self.status['adding_keyframes_stopped']




# Main
if __name__ == '__main__':

# Import libraries
    import cv2
    import g2o

    import os
    import sys
    import argparse

    from threading import Thread
    
    from components import Camera
    from components import StereoFrame
    from feature import ImageFeature
    from params import ParamsKITTI, ParamsEuroc
    from dataset import KITTIOdometry, EuRoCDataset
    

# Define structure for execution arguments in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-viz', action='store_true', help='do not visualize')
    parser.add_argument('--dataset', type=str, help='dataset (KITTI/EuRoC)', 
        default='KITTI')
    parser.add_argument('--path', type=str, help='dataset path', 
        default='path/to/your/KITTI_odometry/sequences/00')
    args = parser.parse_args()

# Define testing datasets
    if args.dataset.lower() == 'kitti':
        params = ParamsKITTI()
        dataset = KITTIOdometry(args.path)
    elif args.dataset.lower() == 'euroc':
        params = ParamsEuroc()
        dataset = EuRoCDataset(args.path)

    sptam = SPTAM(params)

# Define visualization
    visualize = not args.no_viz
    if visualize:
        from viewer import MapViewer
        viewer = MapViewer(sptam, params)

# Define camera
    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, 
        dataset.cam.width, dataset.cam.height, 
        params.frustum_near, params.frustum_far, 
        dataset.cam.baseline)


# Set maximum time for execution of the algorithm
    durations = []
    for i in range(len(dataset))[:100]:
        featurel = ImageFeature(dataset.left[i], params)
        featurer = ImageFeature(dataset.right[i], params)
        timestamp = dataset.timestamps[i]

        time_start = time.time()  

# Set thread and start it
        t = Thread(target=featurer.extract)
        t.start()
        featurel.extract()
        t.join()
   
# Set visual frame     
        frame = StereoFrame(i, g2o.Isometry3d(), featurel, featurer, cam, timestamp=timestamp)

# Initialize algorithm for start
        if not sptam.is_initialized():
            sptam.initialize(frame)
        else:
            sptam.track(frame)

# Track the time of execution
        duration = time.time() - time_start
        durations.append(duration)
        print('duration', duration)
        print()
        print()
        
        if visualize:
            viewer.update()

    print('num frames', len(durations))
    print('num keyframes', len(sptam.graph.keyframes()))
    print('average time', np.mean(durations))

# Visualize graph when algorithm has stopped
    sptam.stop()
    if visualize:
        viewer.stop()