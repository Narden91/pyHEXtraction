import hydra
import os
import pandas as pd
import preprocessing 


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    
    # Get all the folders (complete PATH) inside data (to get only folder name replace f.path with f.name)
    folders_in_data = [f.path for f in os.scandir(config.settings.data_source) if f.is_dir()]
    
    #print(folders_in_data)
    
    for folder in folders_in_data:
        
        images_folder = [f.path for f in os.scandir(folder) if f.is_dir()]
        
        # Process the images in Images folder if necessary
        # preprocessing.process_images_files(images_folder,config.settings.images_extension)
        
        # Get all the .csv files (COMPLETE PATH) inside the folder 
        task_file_list = [f.path for f in os.scandir(folder) if f.is_file() and f.path.endswith(config.settings.file_extension)]
                    
        # Sort the file based on filename
        task_file_list = sorted(task_file_list, key = lambda x: len(x))
        
        # Iterate over the task file in the data folder
        for task_number, task in enumerate(task_file_list):
            
            # Debug compute only the first file in the folder
            if task_number < 1:     
                
                #print(f"Filename: {task} \n")
                
                # Create the dataframe from the csv data
                task_dataframe = preprocessing.load_data_from_csv(task)
                                    
                # Filter the dataframe by the type of points
                #task_dataframe = preprocessing.points_type_filtering(task_dataframe,"onpaper")
                
                print(f"Raw Dataframe: \n {task_dataframe} \n")
                
                # Create the arrays of points from the dataframe columns
                task_dataframe = preprocessing.coordinates_manipulation(task_dataframe)
                
                # Plot with Matplotlib
                # preprocessing.create_array_plot(task_dataframe["PointX"].to_numpy(), task_dataframe["PointY"].to_numpy())
                
                # Create image using points coordinates and pressure
                preprocessing.create_image_from_data(task_dataframe["PointX"].to_numpy(), 
                                                      task_dataframe["PointY"].to_numpy(), 
                                                      task_dataframe["Pressure"].to_numpy())
                
                # # Compute Speed and Acceleration from Points (x,y)
                # velocity, acceleration = preprocessing.compute_speed_and_acceleration(task_dataframe["PointX"].to_numpy(),
                #                                                                       task_dataframe["PointY"].to_numpy())
                
                # # Computer Vel and Acc mean of the array
                # vel_mean = velocity.mean()
                # acc_mean = acceleration.mean()
                
                # print(f"Num of points ({len(velocity)}) \n Velocity Mean: {vel_mean} \n Acceleration Mean: {acc_mean} \n")
                
                # task_dataframe["Velocity"] = pd.Series(velocity)
                # task_dataframe["Acceleration"] = pd.Series(acceleration) 

                print(task_dataframe)
            else:
                break  
               
    return

if __name__== '__main__':
    main()
    
    