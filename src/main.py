import sys

sys.dont_write_bytecode = True

import hydra
import os
import pandas as pd
import preprocessing
import feature_extraction_module


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    # Check data directory
    try:
        # If directory has not yet been created
        os.makedirs(config.settings.data_source)

        sys.exit("Empty directory data")
    except OSError as excep:
        if not os.listdir(config.settings.data_source):
            sys.exit("Empty directory data")
        else:
            # Get all the folders (complete PATH) inside data (to get only folder name replace f.path with f.name)
            folders_in_data = [f.path for f in os.scandir(config.settings.data_source) if f.is_dir()]

    # Check background images directory  
    try:
        # If directory has not yet been created
        os.makedirs(config.settings.background_images_folder)

        sys.exit("Empty background images folder")
    except OSError as excep:
        if not os.listdir(config.settings.background_images_folder):
            sys.exit("Empty background images folder")
        else:
            # Get the background_images
            background_images_list = preprocessing.get_images_in_folder(config.settings.background_images_folder)

    # Iterate over the folders in data
    for folder in folders_in_data:

        # list of the images file in the subject folder
        # images_folder = [f.path for f in os.scandir(folder) if f.is_dir()]

        # Process the images in Images folder if necessary
        # preprocessing.process_images_files(images_folder,config.settings.images_extension)

        # Get all the .csv files (COMPLETE PATH) inside the folder 
        task_file_list = [f.path for f in os.scandir(folder) if
                          f.is_file() and f.path.endswith(config.settings.file_extension)]

        # Sort the file based on filename
        task_file_list = sorted(task_file_list, key=lambda x: len(x))

        # Iterate over the task file in the data folder and the background image
        for task_number, (task, bck_image) in enumerate(zip(task_file_list, background_images_list)):

            # Debug compute only the first "n" file in the folder
            if task_number == 1:
                # Create the dataframe from the csv data
                task_dataframe = preprocessing.load_data_from_csv(task)

                # Filter the dataframe by the type of points (onair / onpaper)
                # task_dataframe = preprocessing.points_type_filtering(task_dataframe,"onpaper")

                # Correct the coordinates system from Digitizer origin -> Image Standard origin
                task_dataframe = preprocessing.coordinates_manipulation(task_dataframe)

                # Show image of the current task created from the csv
                # preprocessing.create_image_from_data(task_dataframe["PointX"].to_numpy(), 
                #                                      task_dataframe["PointY"].to_numpy(), 
                #                                      task_dataframe["Pressure"].to_numpy(),
                #                                      bck_image, task, config)

                # Show image-video of the current task created from the csv
                # preprocessing.create_gif_from_data(task_dataframe["PointX"].to_numpy(), 
                #                                      task_dataframe["PointY"].to_numpy(), 
                #                                      task_dataframe["Pressure"].to_numpy(),
                #                                      bck_image, task, config)

                # Print the csv data of the current task
                print(f"[+] Task Preprocessed: \n {task_dataframe} \n")

                print(f"[+] Starting Feature Extraction \n")

                # Convert the points to strokes
                task_dataframe = feature_extraction_module.convert_from_points_to_strokes(task_dataframe)

            # else:
            #     break  

    return


if __name__ == '__main__':
    main()
