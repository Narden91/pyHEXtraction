import hydra
import pandas as pd
import numpy as np
import re
import os
import csv
import preprocessing 
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import minmax_scale


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    
    # Get all the folders (complete PATH) inside data (to get only folder name replace f.path with f.name)
    folders_in_data = [f.path for f in os.scandir(config.data_source) if f.is_dir()]

    # Debug variable
    count_var = True
    
    for folder in folders_in_data:
        
        # Debug compute only the first folder of the folder list
        if count_var:    
            # Get all the .csv files (COMPLETE PATH) inside the folder 
            task_file_list = [f.path for f in os.scandir(folder) if f.is_file() and f.path.endswith(config.file_extension)]
            
            # Sort the file based on filename
            task_file_list = sorted(task_file_list, key = lambda x: len(x))
            
            for task in task_file_list:
                
                # Debug compute only the first file in the folder
                if count_var:     
                    
                    print(f"Filename: {task} \n")
                    
                    task_dataframe = preprocessing.dataframe_from_csv(task)
                    
                    # print(f"Dataframe: \n {task_dataframe} \n")
                    
                    task_dataframe = preprocessing.points_type_filtering(task_dataframe,"onpaper")
                  
                    print(f"Dataframe: \n {task_dataframe} \n")
                    
                    # Extract column from dataframe and convert to numpy array    
                    x = preprocessing.dataframe_column_to_array(task_dataframe,"PointX")
                    y = preprocessing.dataframe_column_to_array(task_dataframe,"PointY")
                    
                    # Remove the Wacom One offset from points
                    x,y = preprocessing.remove_offset_points(x,y)
                    
                    # Compute Speed and Acceleration from Points (x,y)
                    vel,acc = preprocessing.compute_speed_and_acceleration(x,y)
                    
                    print(f"Num of points ({len(vel)}) \n Velocity: {vel} \n Acceleration: {acc} \n")
                
                    #Canvas dimensions
                    height = 1080 
                    width = 1920
                    thickness = 3
                    color = [255,0,0]
                
                    # Array Scaling with MinMaxScaler
                    # x = np.int_(minmax_scale(x, feature_range=(0,width)))
                    # y = np.int_(minmax_scale(y, feature_range=(0,height)))
                    
                    # Merge 2 array 
                    points = np.column_stack((x, y))
                    
                    print(points)
                    
                    # Create Image Matrix
                    image = np.zeros((height, width, 3), np.uint8)
                    
                    # Fill the Image with white color
                    image.fill(255)
                    
                    # Loop through all the points for drawing on the canvas
                    for i in range(1, len(points)):
                        start = tuple(points[i - 1])
                        end = tuple(points[i])
                        cv2.line(image, start, end, color, thickness)
                    
                    # Show the image in a new Window
                    cv2.imshow("Image", image)
                    cv2.waitKey(0)

  
                    count_var = False
                else:
                    break  
        else:
            break        
    
    return

if __name__== '__main__':
    main()
    
    