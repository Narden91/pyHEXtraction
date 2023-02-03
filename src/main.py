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
    
    #print(folders_in_data)

    # Debug variable
    count_var = True
    
    for folder in folders_in_data:
        
        images_folder = [f.path for f in os.scandir(folder) if f.is_dir()]
        
        # Process the images in Images folder if necessary
        preprocessing.process_images_files(images_folder,config.images_extension)
        
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
                                        
                    task_dataframe = preprocessing.points_type_filtering(task_dataframe,"onpaper")
                  
                    print(f"Dataframe: \n {task_dataframe} \n")
                    
                    # Create the arrays of points from the dataframe columns
                    x,y = preprocessing.process_and_create_arrays_points(task_dataframe)
                    
                    # Create image from array of points
                    preprocessing.create_image_from_array(x,y)
                    
                    # Compute Speed and Acceleration from Points (x,y)
                    vel,acc = preprocessing.compute_speed_and_acceleration(x,y)
                    
                    # Computer Vel and Acc mean of the array
                    vel_mean = vel.mean()
                    acc_mean = acc.mean()
                    
                    print(f"Num of points ({len(vel)}) \n Velocity Mean: {vel_mean} \n Acceleration Mean: {acc_mean} \n")
  
                    count_var = False
                else:
                    break  
        else:
            break        
    
    return

if __name__== '__main__':
    main()
    
    