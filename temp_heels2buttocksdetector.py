from calc_distance import calc_distance

def heels2buttocks_left_up_detector(landmarks):
    """
    Detects the left up pose.
    """
    # if landmarks:
    left_heel_hip_dist = calc_distance(landmarks, 27, 23)
    left_knee_hip_dist = calc_distance(landmarks, 25, 23)
    if left_heel_hip_dist < left_knee_hip_dist:
        return True

    return False

def heels2buttocks_right_up_detector(landmarks):
    """
    Detects the right up pose.
    """
    # if landmarks:
    right_heel_hip_dist = calc_distance(landmarks, 28, 24)
    right_knee_hip_dist = calc_distance(landmarks, 26, 24)
    if right_heel_hip_dist < right_knee_hip_dist:
        return True

    return False

def heels2buttocks_left_down_detector(landmarks):
    """
    Detects the left down pose.
    """
    # if landmarks:
    left_heel_hip_dist = calc_distance(landmarks, 27, 23)
    left_knee_hip_dist = calc_distance(landmarks, 25, 23)
    if left_heel_hip_dist > left_knee_hip_dist * 1.5:
        return True

    return False

def heels2buttocks_right_down_detector(landmarks):
    """
    Detects the right down pose.
    """
    # if landmarks:
    right_heel_hip_dist = calc_distance(landmarks, 28, 24)
    right_knee_hip_dist = calc_distance(landmarks, 26, 24)
    if right_heel_hip_dist > right_knee_hip_dist * 1.5:
        return True

    return False