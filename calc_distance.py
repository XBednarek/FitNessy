def calc_distance_array(array, point1, point2):
    """
    Calculate the Euclidean distance between two landmarks
    """
    # point 1
    point1_y = array[point1][1]
    point1_x = array[point1][0]
    # point 2
    point2_y = array[point2][1]
    point2_x = array[point2][0]

    # y distance
    if point1_y > point2_y:
        dist_y = point1_y - point2_y
    else:
        dist_y = point2_y - point1_y
    # x distance
    if point1_x > point2_x:
        dist_x = point1_x - point2_x
    else:
        dist_x = point2_x - point1_x

    dist = (dist_x**2 + dist_y**2)**0.5

    return dist