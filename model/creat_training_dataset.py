import os
import cv2
import numpy as np
from fitness.tools.py import image2position, MEDIAPIPE_BASE_MODEL,MEDIAPIPE_STATIC_MODEL

def creat_training_dataset(root_folder,usefull_point): 
    """
    Create a training dataset by extracting specific pose landmarks from images stored in labeled subfolders.

    Args:
        root_folder (str): Path to the root folder containing subfolders for each class/label. 
                           Each subfolder should contain images corresponding to that label.
        usefull_point (list[int]): List of indices of pose landmarks to keep for each image. 
                                   For example, [11, 13, 15, 12, 14, 16] for shoulders, elbows, wrists.

    Returns:
        list[list]: A list of rows, where each row is a list containing:
                     - selected_points (np.ndarray): Array of shape (len(usefull_point), 3) 
                       representing the selected landmarks [x, y, z] of the image.
                     - label (str): The class/label of the image.
    """
    data = []
    # Iterate through the dataset
    for label in os.listdir(root_folder):
        # Retrieve all the subfolders
        folder = os.path.join(root_folder, label)
        # if datset empty
        if not os.path.isdir(folder):
            continue

        #iterate through all sub folder
        for file in os.listdir(folder):
            # retrieve all pictured in the folder
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                # retrieve the path
                path = os.path.join(folder, file)
                # load images
                # convert in RGB for mediapipe
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

                try:
                    # retrieve all landmarkers
                    pos_vector = image2position(img,mediapipe_model = MEDIAPIPE_STATIC_MODEL)
                    # chek if pose is none otherwise raise an error
                    if pos_vector is None:
                        # print("----")
                        # print(pos_vector)
                        # print("----")
                        # print(image2position(img, show=True))
                        print(f"No pose detected in image {path}")
                        continue 
                    # select only landmarkers that we need
                    selected_points = np.array(pos_vector)[usefull_point]
                    # link the position with the label
                    row = [selected_points, label]
                    data.append(row)

                except Exception as e:
                    print(f"Erreur image {path}: {e}")
                    continue

    return data