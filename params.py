# Import libraries
import cv2

# Set general parameters
class Params(object):
    def __init__(self):
 
# Set parameter for execution
        self.pnp_min_measurements = 10
        self.pnp_max_iterations = 10
        self.init_min_points = 10

# Set parameters for view
        self.local_window_size = 10
        self.ba_max_iterations = 10

# Set rate for tracking
        self.min_tracked_points_ratio = 0.5

# Set values for THREAD - LOOP CLOSURE
        self.lc_min_inbetween_frames = 10   # frames
        self.lc_max_inbetween_distance = 3  # meters
        self.lc_embedding_distance = 22.0
        self.lc_inliers_threshold = 15
        self.lc_inliers_ratio = 0.5
        self.lc_distance_threshold = 2      # meters
        self.lc_max_iterations = 20

        self.ground = False

        self.view_camera_size = 1

# THREAD - POSE TRACKING
# STEP - MATCHING
# Set parameters for EuRoC dataset (recorded from AscTec Firefly hex-rotor helicopter) - industrial hall
class ParamsEuroc(Params):
    
    def __init__(self, config='GFTT-BRIEF'):
        super().__init__()

# Set descriptor-descriptor pair GFTT-BRIEF for feature detection
        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=15.0, 
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

# Set alternative descriptor pair ORB-BRIEF
        elif config == 'ORB-BRIEF':
            self.feature_detector = cv2.ORB_create(
                nfeatures=200, scaleFactor=1.2, nlevels=1, edgeThreshold=31)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
            
        else:
            raise NotImplementedError

# Set bruce force matcher of cv2 defined with hamming distance and return all values not only best match,
# therefore crossCheck = false
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Set visual aspects for matching
        self.matching_cell_size = 15   # pixels
        self.matching_neighborhood = 2
        self.matching_distance = 25

# Frustum projection of the map points around the predicted pose
        self.frustum_near = 0.1  # meters
        self.frustum_far = 50.0

# Set values for distances to detect valid ones
        self.lc_max_inbetween_distance = 4   # meters
        self.lc_distance_threshold = 1.5
        self.lc_embedding_distance = 22.0

        self.view_image_width = 400
        self.view_image_height = 250
        self.view_camera_width = 0.1
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -1
        self.view_viewpoint_z = -10
        self.view_viewpoint_f = 2000

    
        
# Set parameter for KITTI dataset (recorded on KITTI Vision Benchmark Suite) - street
class ParamsKITTI(Params):
    def __init__(self, config='GFTT-BRIEF'):
        super().__init__()

# Set descriptor-descriptor pair GFTT-BRIEF for feature detection
        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=12.0, 
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

# Set alternative descriptor pair ORB-BRIEF
        elif config == 'GFTT-BRISK':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=2000, minDistance=15.0, 
                qualityLevel=0.01, useHarrisDetector=False)

            self.descriptor_extractor = cv2.BRISK_create()

        elif config == 'ORB-ORB':
            self.feature_detector = cv2.ORB_create(
                nfeatures=1000, scaleFactor=1.2, nlevels=1, edgeThreshold=31)
            self.descriptor_extractor = self.feature_detector

        else:
            raise NotImplementedError

# Set bruce force matcher of cv2 defined with hamming distance and return all values not only best match,
# therefore crossCheck = false
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Set visual aspects for matching
        self.matching_cell_size = 15   # pixels
        self.matching_neighborhood = 3
        self.matching_distance = 30

# Frustum projection of the map points around the predicted pose
        self.frustum_near = 0.1    # meters
        self.frustum_far = 1000.0

        self.ground = True

# Set values for distances to detect valid ones
        self.lc_max_inbetween_distance = 50
        self.lc_distance_threshold = 15
        self.lc_embedding_distance = 20.0

        self.view_image_width = 400
        self.view_image_height = 130
        self.view_camera_width = 0.75
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -500   # -10
        self.view_viewpoint_z = -100   # -0.1
        self.view_viewpoint_f = 2000