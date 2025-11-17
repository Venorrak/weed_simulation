import cv2
import numpy as np
import math
import time
import argparse

import funcs
import video

# Hardware constants (never change during runtime)
NUMBER_OF_COLS: int = 8  # number of solenoids
SPRAY_INTENSITY: int = 100  # intensity of the spray 0-255

# Simulation constants
MAX_SPEED: float = 2.5  # in km/h simulated
MIN_SPEED: float = 0.5  # in km/h simulated
OUTPUT_FPS: float = 20  # fps of the output video

def get_parameters():
    """Get simulation parameters from user or use defaults."""
    print("Do you want to use the default parameters? (y/n)")
    change_params = input()
    
    if change_params == "n":
        print("Enter the solenoid activation threshold (0 - 100):")
        threshold = int(input())
        print("Enter the minimum number of frames a frame will be displayed (x>=1)(dependent on speed):")
        min_fps = int(input())
        print("Enter the maximum number of frames a frame will be displayed (dependent on speed):")
        max_fps = int(input())
        print("Enter the factor to resize the frame (0-1 float):")
        size_factor = float(input())
        print("Enter the range of the spray in px (Default : 250):")
        spray_range = int(int(input()) * size_factor)
    else:
        print("Default parameters will be used")
        threshold = 1
        min_fps = 1
        max_fps = 5
        size_factor = 0.5
        spray_range = int(300 * size_factor)
    
    # Calculate dependent parameters
    row_px_from_top = int(200 * size_factor)
    font_scale = size_factor
    spray_spacing = int(40 * size_factor)
    
    return threshold, min_fps, max_fps, size_factor, spray_range, row_px_from_top, font_scale, spray_spacing

# params for corner detection 
FEATURE_PARAMS = dict( maxCorners = 100, 
                       qualityLevel = 0.3, 
                       minDistance = 7, 
                       blockSize = 7 ) 
  
# Parameters for lucas kanade optical flow
LK_PARAMS = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

#to rotate the image according to the planning at the given timestamp
PLANNING: list = [
    {
        "timestamp": 18,
        "rotation": 270
    },
    {
        "timestamp": 100,
        "rotation": 90
    }
]

def analyze_frame(cap, params, frame_state):
    """
    Analyze a single frame from the video.
    
    Args:
        cap: Video capture object
        params: Dictionary with simulation parameters
        frame_state: Dictionary with mutable state (frame counters, totals, etc.)
    
    Returns:
        tuple: (result_frame, speed)
    """
    # Extract parameters
    size_factor = params['size_factor']
    threshold = params['threshold']
    spray_range = params['spray_range']
    row_px_from_top = params['row_px_from_top']
    font_scale = params['font_scale']
    spray_spacing = params['spray_spacing']
    source_fps = params['source_fps']
    planning = params['planning']
    feature_params = params['feature_params']
    
    # Get the start time of current frame
    start_frame = time.time()

    # declare array of false for solenoid sim
    solenoid_active = funcs.declare_solenoid_active(NUMBER_OF_COLS)

    # Get the frame from the video
    ret, frame_original = cap.read()

    # Rotate the frame according to the planning
    frame_state['current_frame'] += 1
    frame = funcs.rotate(frame_original, frame_state['current_frame'], source_fps, planning)
    
    ##################
    ## optical flow ##
    ##################
    
    #calculate the optical flow
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, status, err = cv2.calcOpticalFlowPyrLK(analyze_frame.old_gray, frame_gray, analyze_frame.p0, None)
    
    # Select good points 
    good_new = p1[status == 1] 
    good_old = analyze_frame.p0[status == 1]
    
    # Preallocate the array to avoid repeated allocation in the loop
    all_movements = np.zeros((len(good_new), 2))
        
    # Using NumPy operations for vectorized computation
    a_b = good_new.reshape(-1, 2)
    c_d = good_old.reshape(-1, 2)

    # Calculate movements
    all_movements = a_b - c_d    
    
    #remove vectors that are not in the same general direction
    all_angles = []
    for i in range(len(all_movements)):
        if all_movements[i][0] == 0:
            all_movements[i][0] = 0.0001
        vector = [all_movements[i][0], all_movements[i][1]]
        angle = np.arctan(vector[1] / vector[0])
        degree = np.degrees(angle)
        all_angles.append(degree)
    median_angle = np.median(np.sort(all_angles))
    for i in range(len(all_angles)):
        if all_angles[i] > median_angle + 5 or all_angles[i] < median_angle - 5:
            np.delete(all_movements, i)
    
    # Calculate the average movement
    delta_movement = (np.mean(all_movements, axis=0).astype(int) * size_factor).astype(int)
    
    # Prepare next optical flow calculation
    analyze_frame.old_gray = frame_gray.copy() 
    analyze_frame.p0 = cv2.goodFeaturesToTrack(analyze_frame.old_gray, mask=None, **feature_params)
    
    # Display the optical flow
    if (False): 
        black_screen = np.zeros_like(frame)
            
        # Draw lines and circles on the black screen
        for (a, b), (c, d) in zip(a_b, c_d):
            a, b = int(a), int(b)
            c, d = int(c), int(d)
            cv2.line(black_screen, (a, b), (c, d), (255, 255, 255), 2)
            cv2.circle(black_screen, (a, b), 2, 155, -1)
            
        cv2.imshow("Optical flow", black_screen)

    #########################
    ## end of optical flow ##
    #########################
    
    # resize the frame
    frame = cv2.resize(frame, (0, 0), fx=size_factor, fy=size_factor)

    # get the excess of green 
    exg = funcs.calcluate_exg(frame)
    
    # threshold the image
    tresh = funcs.threshold(exg)

    # open and close the image
    opened_closed = funcs.open_and_close_image(tresh)

    # color the detected green (weeds) on the original frame
    frame[opened_closed==255] = (0,255,0)

    # get the active solenoids and the speed the robot should move
    solenoid_active = funcs.activate_solenoids(NUMBER_OF_COLS, row_px_from_top,
                                                opened_closed, solenoid_active, 
                                                threshold)
    
    # create mask of the sprayed weed
    sprayed = funcs.get_sprayed_weed(NUMBER_OF_COLS, row_px_from_top, opened_closed, solenoid_active,
                                     spray_range, delta_movement, SPRAY_INTENSITY, spray_spacing)

    #color the sprayed weed on the original frame
    for i in range(1, 256):
        frame[sprayed == i] = (0, 255-i, i)
        
    ##################
    ##     stats    ##
    ##################
    
    #print("Delta movement: ", delta_movement)
    last_sprayed_frame = analyze_frame.old_spray
    height, width, _ = last_sprayed_frame.shape
    
    #get the parts of the last sprayed frame that are not visible in the current frame
    if delta_movement[0] < 0:
        #robot moved to the right
        h_part = last_sprayed_frame[:, 0:-delta_movement[0]]
        
        if delta_movement[1] < 0:
            #robot moved up
            v_part = last_sprayed_frame[height+delta_movement[1]:, -delta_movement[0]:]
        elif delta_movement[1] > 0:
            #robot moved down
            v_part = last_sprayed_frame[height-delta_movement[1]:, -delta_movement[0]:]
        else:
            #print("Robot did not move vertically")
            v_part = None
        
    elif delta_movement[0] > 0:
        #robot moved to the left
        h_part = last_sprayed_frame[:, width-delta_movement[0]:]
        
        if delta_movement[1] < 0:
            #robot moved up
            v_part = last_sprayed_frame[height+delta_movement[1]:, 0:width-delta_movement[0]]
        elif delta_movement[1] > 0:
            #robot moved down
            v_part = last_sprayed_frame[height-delta_movement[1]:, 0:width-delta_movement[0]]
        else:
            #print("Robot did not move vertically")
            v_part = None
        
    else:
        #print("Robot did not move horizontally")
        h_part = None
        
        if delta_movement[1] < 0:
            #robot moved up
            v_part = last_sprayed_frame[height+delta_movement[1]:, :]
        elif delta_movement[1] > 0:
            #robot moved down
            v_part = last_sprayed_frame[height-delta_movement[1]:, :]
        else:
            #print("Robot did not move vertically")
            v_part = None
        
    #get the total red and green in the parts
    if h_part is None:
        h_red = 0
        h_green = 0
    else:
        #cv2.imshow("h_part", h_part)
        cv2.waitKey(1)  # Add a delay of 1 millisecond to allow time for the frame to be displayed
        h_red = np.sum(h_part[:, :, 2])
        h_green = np.sum(h_part[:, :, 1])
        
    if v_part is None:
        v_red = 0
        v_green = 0
    else:
        #cv2.imshow("v_part", v_part)
        cv2.waitKey(1)  # Add a delay of 1 millisecond to allow time for the frame to be displayed
        v_red = np.sum(v_part[:, :, 2])
        v_green = np.sum(v_part[:, :, 1])
    
    # add the total red and green to the global total
    total_red = np.add(h_red, v_red)
    total_green = np.add(h_green, v_green)
    # print("Total red: ", total_red)
    # print("Total green: ", total_green)
    frame_state['global_total_red'] = np.add(total_red, frame_state['global_total_red'])
    frame_state['global_total_green'] = np.add(total_green, frame_state['global_total_green'])
    
    #save the sprayed frame for the next frame
    blank_canvas = np.zeros_like(frame)
    blank_canvas[opened_closed==255] = (0,255,0)
    for i in range(1, 256):
        blank_canvas[sprayed == i] = (0, 255-i, i)
    #cv2.imshow("Sprayed", blank_canvas)
    analyze_frame.old_spray = blank_canvas
    
    ##################
    ##   end stats  ##
    ##################

    # get the speed of the robot
    speed = funcs.get_speed(solenoid_active, MAX_SPEED, MIN_SPEED)

    # calculate the fps
    end_frame = time.time()
    delta_time_frame = end_frame - start_frame
    fps = 1 / delta_time_frame

    # print the UI on the frame
    result = funcs.printUI(frame, NUMBER_OF_COLS, row_px_from_top,
                            solenoid_active, fps, speed, frame_state['current_frame'],
                            font_scale, 1)
    

    return result, speed

def main():
    """Main entry point for the weed simulation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Weed simulation video analyzer")
    parser.add_argument("--video-path", type=str, default=None, help="Path to the video file to open")
    args = parser.parse_args()

    import os
    import json
    
    # Load or create config
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {"last_video_path": None}

    # Determine video path: CLI > config > default
    if args.video_path:
        video_path = args.video_path
    elif config.get("last_video_path"):
        video_path = config["last_video_path"]
    else:
        video_path = "test_data/video1.mp4"

    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        print("Please provide a valid path with --video-path.")
        return

    # Save last used path to config
    config["last_video_path"] = video_path
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Get simulation parameters from user
    threshold, min_fps, max_fps, size_factor, spray_range, row_px_from_top, font_scale, spray_spacing = get_parameters()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video device: {video_path}")
        return
        
    source_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create parameter dictionary
    params = {
        'size_factor': size_factor,
        'threshold': threshold,
        'spray_range': spray_range,
        'row_px_from_top': row_px_from_top,
        'font_scale': font_scale,
        'spray_spacing': spray_spacing,
        'source_fps': source_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'planning': PLANNING,
        'feature_params': FEATURE_PARAMS
    }
    
    # Create frame state dictionary
    frame_state = {
        'current_frame': 1,
        'global_total_red': 0,
        'global_total_green': 0
    }
    
    # Sample first frame to get the width and height for the output video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * size_factor)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * size_factor)
    video.open_video(width, height, OUTPUT_FPS)
    
    # Initialize the optical flow variables
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return
        
    analyze_frame.old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    analyze_frame.old_gray = cv2.rotate(analyze_frame.old_gray, cv2.ROTATE_90_CLOCKWISE)
    analyze_frame.p0 = cv2.goodFeaturesToTrack(analyze_frame.old_gray, mask=None, **FEATURE_PARAMS)
    
    # Initialize the old_spray with the first frame
    sample_frame = old_frame.copy()
    sample_frame = cv2.rotate(sample_frame, cv2.ROTATE_90_CLOCKWISE)
    sample_frame = cv2.resize(sample_frame, (0, 0), fx=size_factor, fy=size_factor)
    analyze_frame.old_spray = np.zeros_like(sample_frame)
    
    # Analyze the first frame
    if cap.isOpened():
        result = analyze_frame(cap, params, frame_state)

    done = False
    
    while cap.isOpened():
        try:
            result, speed = analyze_frame(cap, params, frame_state)
            
            if not result.any():
                done = True
            
            # Add the frame to the output video
            video.write_frame(result, OUTPUT_FPS, speed, MAX_SPEED, MIN_SPEED, max_fps, min_fps)
            cv2.imshow("Result", result)
            
        except Exception as e:
            print(e)
            video.close_video()
            # Calculate the efficiency
            efficiency = funcs.calculate_efficiency(frame_state['global_total_green'], frame_state['global_total_red'])
            print("Efficiency: ", efficiency, "%")
            break
        
        # Press q to close the window
        if cv2.waitKey(1) & 0xFF == ord('q') or done:
            # Calculate the efficiency
            efficiency = funcs.calculate_efficiency(frame_state['global_total_green'], frame_state['global_total_red'])
            print("Efficiency: ", efficiency, "%")
            cap.release()
            video.close_video()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()