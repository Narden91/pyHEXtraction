import sys

sys.dont_write_bytecode = True

import hydra
import os
import pandas as pd
import preprocessing
from data_handler import check_directories_and_load_data, load_anagrafica, read_anagrafica, save_data_to_csv
from data_conversion_module import convert_to_HandwritingSample_library, stroke_segmentation
from feature_extraction_module import stroke_approach_feature_extraction


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    # Check data directory and retrieve data
    folders_in_data, background_images_list = check_directories_and_load_data(config)

    # Iterate over the folders in data
    for folder in folders_in_data:
        # Load the anagrafica file
        anagrafica_file = load_anagrafica(folder)

        # Read the anagrafica file
        anagrafica_data = read_anagrafica(anagrafica_file)

        # list of the image file in the subject folder
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
            # Debug: computes only the "n" file in the folder
            if task_number == 0:
                # -------------------Preprocessing Section------------------- #
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
                # print(f"[+] Task Preprocessed: \n {task_dataframe.head().to_string()} \n")

                # -------------------Library Conversion Section------------------- #
                # Manipulate the dataframe to be ready for HandwritingSample library
                task_dataframe = convert_to_HandwritingSample_library(task_dataframe)

                # Debug: print the data after the transformation
                # print(f"[+] Data after Transformation for HandwritingSample Library : \n{task_dataframe}")

                # Get the strokes using the HandwritingSample library
                stroke_list = stroke_segmentation(task_dataframe)

                # -------------------Feature Extraction Section------------------- #
                # Stroke Approach feature extraction
                stroke_approach_dataframe = stroke_approach_feature_extraction(stroke_list, task_dataframe)

                # print(f"[+] Stroke Approach Features: \n{stroke_approach_dataframe.to_string()}")

                # Save the features extracted from the current task
                save_data_to_csv(stroke_approach_dataframe, task_number + 1, folder, anagrafica_data, config)
    return 0


if __name__ == '__main__':
    main()
