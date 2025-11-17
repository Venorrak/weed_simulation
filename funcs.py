import cv2
import numpy as np
import time
from functools import wraps

# Global timing storage
_function_timings = {}

def timed_function(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        # Store timing
        if func.__name__ not in _function_timings:
            _function_timings[func.__name__] = []
        _function_timings[func.__name__].append(elapsed)
        
        return result
    return wrapper

def get_timing_report():
    """Generate a timing report for all timed functions."""
    if not _function_timings:
        return "No timing data available."
    
    report = "\n" + "="*70 + "\n"
    report += "FUNCS.PY TIMING REPORT\n"
    report += "="*70 + "\n"
    
    total_all = 0
    stats = []
    
    for func_name, times in _function_timings.items():
        count = len(times)
        total = sum(times)
        avg = total / count
        min_t = min(times)
        max_t = max(times)
        
        total_all += total
        stats.append((func_name, count, avg, min_t, max_t, total))
    
    # Sort by total time (descending)
    stats.sort(key=lambda x: x[5], reverse=True)
    
    for func_name, count, avg, min_t, max_t, total in stats:
        percentage = (total / total_all * 100) if total_all > 0 else 0
        report += f"\n{func_name}:\n"
        report += f"  Calls: {count:6d}  |  Avg: {avg*1000:7.2f} ms  |  Min: {min_t*1000:6.2f} ms  |  Max: {max_t*1000:6.2f} ms\n"
        report += f"  Total: {total*1000:7.2f} ms ({percentage:5.1f}%)\n"
    
    report += "-"*70 + "\n"
    report += f"TOTAL TIME IN FUNCS: {total_all*1000:.2f} ms\n"
    report += "="*70 + "\n"
    
    return report

def reset_timing_data():
    """Reset all timing data."""
    global _function_timings
    _function_timings = {}

@timed_function
def rotate(frame: np.array, current_frame: int, source_fps: int, planning: list):
    """
    rotate the frame according to the planning
    :param frame: The frame to rotate
    :param current_frame: The current frame
    :param source_fps: The source fps of the video
    :param planning: The planning of the rotation
    :return: The rotated frame
    """
    current_second = int(current_frame / source_fps)
    for stage in planning:
        if stage["timestamp"] > current_second:
            rotation = stage["rotation"]
            break
    match rotation:
        case 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        case 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        case 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        case 0:
            frame = frame
    return frame

@timed_function
def divide_frame(frame: np.array, cols: int, row_px: int) -> list[np.array]:
    """
    Divide a frame into rows*cols smaller frames
    :param frame: The frame to divide
    :param cols: The number of columns to divide the frame into
    :param row_px: The number of pixels from the top to divide the frame
    :return: A list of smaller frames
    """

    # Get the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]

    #divide frame in two at a defined number of pixels from the top
    frame = frame[0:int(row_px), 0:frame_width]

    # Calculate the height and width of each smaller frame
    col_width = frame_width // cols
    for i in range(cols):
        divided_frames = [frame[:, i * col_width:(i + 1) * col_width] for i in range(cols)]


    return divided_frames

@timed_function
def get_rgb_channels(img: np.array) -> tuple[np.array, np.array, np.array]:
    """
    Split an image into its channels and return them in RGB order
    :param img: The image to split the channels of
    :return: The channels in RGB order
    """

    # Split the image into channels and convert to float to avoid overflow
    blue_channel, green_channel, red_channel = cv2.split(np.array(img, dtype=np.float32))

    # Return the channels in RGB order (Makes it easier to change channel order if the camera uses a different order)
    return red_channel, green_channel, blue_channel

@timed_function
def calcluate_exg(img) -> np.array:
    """
    Calculat ExG (Excess of Green) using the formula 2 * G - (R + B)
    :param img: The image to calculate ExG on
    :return: The ExG image (grayscale)
    """

    # Split the image into channels and convert to float to avoid overflow
    # BEWARE of the order of the channels
    blue_channel, green_channel, red_channel = cv2.split(np.array(img, dtype=np.float32))

    # Calculate the excess of green
    exg_img = 2 * green_channel - (red_channel + blue_channel)

    # Normalize the image to 0-255
    return cv2.normalize(exg_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # We use CV_8U to save space

@timed_function
def threshold(img: np.array) -> np.array:
    """
    Use Otsu's thresholding on an image
    :param img: The image to blur and threshold
    :return: The thresholded image
    """

    # Apply gaussian blur to the image, then apply Otsu's thresholding
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, treated = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return treated

@timed_function
def open_and_close_image(img: np.array) -> np.array:
    """
    Open and close an image to remove noise (when closing, dilate more to make blobs bigger)
    :param img: The image to open and close
    :return: The opened and closed image
    """

    img = open_image(img)
    img = close_image(img)

    return img

@timed_function
def close_image(img: np.array) -> np.array:
    """
    Close an image to remove noise (dilate then erode) This will help us obtain blobs of vegetation
    :param img: The image to close
    :return: The closed image
    """

    kernel = np.ones((5, 5), np.uint8)

    # Close the image to make the blobs of vegetation bigger
    for i in range(4):
        img = cv2.dilate(img, kernel, iterations=2)
        img = cv2.erode(img, kernel, iterations=1)

    return img

@timed_function
def open_image(img: np.array) -> np.array:
    """
    Open an image to remove noise (erode then dilate)
    :param img: The image to open
    :return: The opened image
    """

    kernel = np.ones((5, 5), np.uint8)

    # Open the image to remove noise
    for i in range(4):
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=2)

    return img

@timed_function
def detect_edges_canny(img: np.array) -> np.array:
    """
    Detect edges using Canny edge detection, Expecting a grayscale image
    :param img: The image to detect edges on
    :return: The image with the edges applied
    """

    # Apply Canny edge detection to the image
    return cv2.Canny(img, 100, 200)

def declare_solenoid_active(cols: int) -> list[bool]:
    """
    Declare the solenoids as active or inactive
    :param cols: The number of columns to declare
    :return: A list of bools representing the solenoids
    """

    # Declare the solenoids as active
    return [False] * cols

@timed_function
def activate_solenoids(cols: int, row_px: int, frame: np.array, solenoid_active: list[bool], threshold: int) -> list[bool]:
    """
    Activate the solenoids if a column is more than % white
    :param cols: The number of columns to activate
    :param frame: The frame to check
    :param solenoid_active: The current state of the solenoids (0-100% of white)
    :param threshold: The threshold in % to activate the solenoids
    :return: The new state of the solenoids
    """

    # Get the divided frames
    divided_frames = divide_frame(frame, cols * 2, row_px)

    # # Check each column
    # for i in range(cols * 2):
    #     # Calculate the percentage of white in the column
    #     white_percentage = 100 - ((np.sum(divided_frames[i] == 0) / divided_frames[i].size) * 100)
    frames_white_percentage = [100 - ((np.sum(divided_frames[i] == 0) / divided_frames[i].size) * 100) for i in range(cols * 2)]
        
    for i in range(cols * 2):
        # Activate the solenoid if the percentage is above the threshold
        if frames_white_percentage[i] > threshold:
            solenoid_active[i // 2] = True
        if i % 2 != 0:
            total_white = (frames_white_percentage[i] + frames_white_percentage[i - 1]) / 2
            if total_white > threshold:
                solenoid_active[i // 2] = True
    return solenoid_active

@timed_function
def get_speed(solenoid_active: list[bool], max_speed: float, min_speed) -> float:
    """
    Get the speed the robot should move
    :param solenoid_active: The active solenoids
    :param max_speed: The maximum speed the robot should move
    :param min_speed: The minimum speed the robot should move
    :return: The speed the robot should move
    """
    #default speed is max speed
    speed = max_speed
    delta_speed = max_speed - min_speed

    if np.sum(solenoid_active) > 1:
        #speed = max_speed - ((activated_solenoids / total_solenoids) * (max_speed - min_speed)
        speed = max_speed - ((np.sum(solenoid_active) / len(solenoid_active)) * delta_speed)
    
    return speed

@timed_function
def printUI(frame: np.array, cols: int, row_px: int, solenoid_active: list[bool], fps: float, speed: float, current_frame : int = -1, font_scale : float = 1, font_thickness : int = 2) -> np.array:
    """
    Print the UI to the frame
    :param frame: The frame to print the UI to
    :param cols: The number of columns
    :param row_px: The number of pixels from the top where the spray is
    :param solenoid_active: The active solenoids
    :param fps: The frames per second of the simulation
    :param speed: The speed the robot should move
    :param current_frame: The current frame of the video
    :param font_scale: The scale of the font
    :param font_thickness: The thickness of the font
    :return: The frame with the UI
    """
    frame_height, frame_width = frame.shape[:2]
    col_width = frame_width // cols
    font_size = 2 * font_scale
    
    for i in range(cols):
        x = int(i * col_width)
        x2 = int(x + (col_width / 2))
        cv2.line(frame, (x, 0), (x, frame_height), (0, 0, 255), font_thickness)
        cv2.line(frame, (x2, 0), (x2, row_px), (0, 0, 100), font_thickness)
        cv2.putText(frame, f"{'ON' if solenoid_active[i] else 'OFF'}", (i * col_width + 30, int(30 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)
    cv2.line(frame, (0, row_px), (frame_width, row_px), (0, 0, 255), font_thickness)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, int(100 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)
    cv2.putText(frame, f"Speed: {speed:.2f}", (10, int(150* font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)
    cv2.putText(frame, f"pression 1/{np.sum(solenoid_active)}", (10, int(200* font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)
    
    if (current_frame != -1):
        cv2.putText(frame, f"Frame: {current_frame}", (10, int(250* font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)

    return frame

def old_get_sprayed_weed(cols:int, row_px: int, frame:np.array, solenoid_active: list[bool], spray_range:int, delta_movement: tuple[int, int], spray_intensity: int) -> np.array:
    global spray_history
    """
    create black frame where only the green in solenoid_active is shown
    :param cols: The number of columns
    :param row_px: The number of pixels from the top to apply the effect
    :param frame: The frame to apply the effect to
    :param solenoid_active: The active solenoids
    :param spray_range: The range of the spray
    :param delta_movement: The movement of the frame
    :return: The frame with the spray effect
    """
    frame_height, frame_width = frame.shape[:2]
    col_width = frame_width // cols
        
    # Delete centers that are too close to each other
    for a in range(len(spray_history)):
        for b in range(a + 1, len(spray_history)):
            if np.sqrt((spray_history[a][0] - spray_history[b][0]) ** 2 + (spray_history[a][1] - spray_history[b][1]) ** 2) < 40:
                spray_history.pop(b)
                break
            
    for center in spray_history:
        # Remove the spray if it's out of bounds
        if center[0] <= 0 or center[0] >= frame_width or center[1] <= 0 or center[1] >= frame_height:
            spray_history.remove(center)
        
    for i in range(len(spray_history)):
        # Move the spray
        spray_history[i] = (spray_history[i][0] + int(delta_movement[0]), spray_history[i][1] + int(delta_movement[1]))

    for i in range(cols):
        #add new spray
        if solenoid_active[i]:
            center = (int(col_width / 2 + (col_width * i)), int(row_px // 2))
            spray_history.append(center)
    
    final= np.zeros_like(frame)
    for center in spray_history:
        black_screen = np.zeros_like(frame)
        for r in range(spray_range):
            intensity = spray_intensity - ((r * spray_intensity) / spray_range)
            cv2.circle(black_screen, center, r, intensity, 2)
            
        min_x = int(center[0]) - spray_range
        max_x = int(center[0]) + spray_range
        min_y = int(center[1]) - spray_range
        max_y = int(center[1]) + spray_range
        if min_x < 0:
            min_x = 0
        if max_x > frame_width:
            max_x = frame_width
        if min_y < 0:
            min_y = 0
        if max_y > frame_height:
            max_y = frame_height
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if frame[y, x] == 0:
                    black_screen[y, x] = 0
        cv2.addWeighted(final, 1, black_screen, 1, 0, final)
    return final

# FIXME : This function is getting slower and slower as time goes on. Need to optimize spray history management.
@timed_function
def get_sprayed_weed(cols: int, row_px: int, frame: np.array, solenoid_active: list[bool], spray_range: int, delta_movement: tuple[int, int], spray_intensity: int, spray_spacing: int) -> np.array:
    if not hasattr(get_sprayed_weed, "spray_history"):
        get_sprayed_weed.spray_history = []
    spray_history = get_sprayed_weed.spray_history
    
    """
    Create a mask representing all sprayed weeds
    :param cols: The number of columns
    :param row_px: The number of pixels from the top to apply the effect
    :param frame: The frame to apply the effect to
    :param solenoid_active: The active solenoids
    :param spray_range: The range of the spray
    :param delta_movement: The movement of the frame
    :param spray_intensity: The intensity of the spray
    :param spray_spacing: The spacing between sprays
    :return: The mask of the sprays
    """
    frame_height, frame_width = frame.shape[:2]
    col_width = frame_width // cols

    # Convert spray_history to a NumPy array for faster operations
    if spray_history:
        spray_history_np = np.array(spray_history)

        # Delete centers that are too close to each other
        dist = np.sqrt(np.sum((spray_history_np[:, np.newaxis] - spray_history_np[np.newaxis, :])**2, axis=-1))
        mask = np.triu(dist < spray_spacing, 1).any(axis=0)
        spray_history_np = spray_history_np[~mask]

        # Remove sprays out of bounds
        in_bounds_mask = (0 < spray_history_np[:, 0]) & (spray_history_np[:, 0] < frame_width) & (0 < spray_history_np[:, 1]) & (spray_history_np[:, 1] < frame_height)
        spray_history_np = spray_history_np[in_bounds_mask]

        # Move the sprays
        spray_history_np += delta_movement

    else:
        spray_history_np = np.empty((0, 2), int)

    # Add new sprays
    new_sprays = np.array([(int(col_width / 2 + (col_width * i)), int(row_px // 2)) for i in range(cols) if solenoid_active[i]])
    if new_sprays.size > 0:
        spray_history_np = np.vstack((spray_history_np, new_sprays))

    # Update spray_history with the modified array
    spray_history = spray_history_np.tolist()

    # Prepare the final and black_screen frames
    final = np.zeros_like(frame)

    for center in spray_history:
        black_screen = np.zeros_like(frame)
        for r in range(spray_range):
            intensity = spray_intensity - ((r * spray_intensity) / spray_range)
            cv2.circle(black_screen, tuple(center), r, intensity, 2)

        min_x, max_x = max(0, int(center[0]) - spray_range), min(frame_width, int(center[0]) + spray_range)
        min_y, max_y = max(0, int(center[1]) - spray_range), min(frame_height, int(center[1]) + spray_range)

        black_screen[min_y:max_y, min_x:max_x][frame[min_y:max_y, min_x:max_x] == 0] = 0

        cv2.addWeighted(final, 1, black_screen, 1, 0, final)
    
    get_sprayed_weed.spray_history = spray_history

    return final

def average_movement(all_movements: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Calculate the average movement from a list of movements
    :param all_movements: The list of movements
    :return: The average movement
    """

    # Calculate the average movement
    return tuple(map(lambda x: sum(x) // len(all_movements), zip(*all_movements)))

def calculate_efficiency(total_green: int, total_red: int) -> float:
    """
    Calculate the efficiency of the spraying
    :param total_green: The total green pixels
    :param total_sprayed: The total sprayed pixels
    :return: The efficiency of the spraying
    """

    total = total_green + total_red
    
    return (total_red * 100) / total if total != 0 else 0