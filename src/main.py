import sys

sys.dont_write_bytecode = True

import hydra
import os
from tqdm import tqdm
from data_processor import DataProcessor
import pandas as pd
import preprocessing
from data_conversion_module import convert_to_HandwritingSample_library, stroke_segmentation
from feature_extraction_module import stroke_approach_feature_extraction, statistical_feature_extraction


class MainClass:
    def __init__(self, config):
        self.config = config
        self.processor = DataProcessor(config)

    def run(self):
        data_source = self.processor.format_path_for_os(self.config.settings.data_source)
        background_images_folder = self.processor.format_path_for_os(self.config.settings.background_images_folder)

        folders_in_data, message = self.processor.get_directory_contents(data_source, 'folders')
        if message:
            print(message)
            return

        background_images_list = []
        if self.config.settings.use_images:
            background_images_list, image_message = self.processor.get_directory_contents(background_images_folder,
                                                                                          'files')
            if image_message:
                print(image_message)
                return

        for folder in tqdm(folders_in_data, desc="Processing folders"):
            # Load the anagrafica file
            anagrafica_file = self.processor.load_anagrafica(folder)
            if not anagrafica_file:
                print(f"[+] No anagrafica file found in {folder}. Skipping.")
                continue

            # Read the anagrafica file
            anagrafica_data = self.processor.read_anagrafica(anagrafica_file)
            if not anagrafica_data:
                print(f"Error reading or invalid anagrafica file format in {folder}. Skipping.")
                continue

            task_file_list = sorted(
                [f.path for f in os.scandir(folder) if
                 f.is_file() and f.path.endswith(self.config.settings.file_extension)],
                key=lambda x: len(x)
            )

            for task_number, task in enumerate(task_file_list):
                # Debug: computes only the "n" file in the folder
                if task_number == 0:
                    # -------------------Preprocessing Section------------------- #
                    task_dataframe = self.processor.load_and_process_csv(task)

                    # Filter the dataframe by the type of points (onair / onpaper)
                    # task_dataframe = preprocessing.points_type_filtering(task_dataframe,"onpaper")

                    # Correct the coordinates system from Digitizer origin -> Image Standard origin
                    task_dataframe = preprocessing.coordinates_manipulation(task_dataframe)

                    if self.config.settings.use_images:
                        self.processor.process_images(folder, self.config.settings.images_extension,
                                                      background_images_list)
                        # Show image of the current task created from the csv
                        preprocessing.create_image_from_data(task_dataframe["PointX"].to_numpy(),
                                                             task_dataframe["PointY"].to_numpy(),
                                                             task_dataframe["Pressure"].to_numpy(),
                                                             background_images_list, task, self.config)
                        # Show image-video of the current task created from the csv
                        preprocessing.create_gif_from_data(task_dataframe["PointX"].to_numpy(),
                                                           task_dataframe["PointY"].to_numpy(),
                                                           task_dataframe["Pressure"].to_numpy(),
                                                           background_images_list, task, self.config)

                    # Print the csv data of the current task
                    print(f"[+] Task After Coordinates manipulation: \n {task_dataframe.head(10).to_string()} \n")

                    # -------------------Library Conversion Section------------------- #
                    # Manipulate the dataframe to be ready for HandwritingSample library
                    task_dataframe = convert_to_HandwritingSample_library(task_dataframe)

                    # Debug: print the data after the transformation
                    print(f"[+] Data after Transformation for HandwritingSample Library : \n"
                          f"{task_dataframe.head(10).to_string()} \n")

                    # Get the strokes using the HandwritingSample library
                    # stroke_list = stroke_segmentation(task_dataframe)

                    # Get the features using the HandwritingSample library from the task
                    handwriting_feature_dataframe = statistical_feature_extraction(task_dataframe)

                    print(f"[+] Handwriting Features: \n{handwriting_feature_dataframe}")

                    # -------------------Feature Extraction Section------------------- #
                    # Stroke Approach feature extraction
                    # stroke_approach_dataframe = stroke_approach_feature_extraction(stroke_list, task_dataframe)

                    # print(f"[+] Stroke Approach Features: \n{stroke_approach_dataframe.to_string()}")

                    # Save the features extracted from the current task
                    # save_data_to_csv(stroke_approach_dataframe, task_number + 1, folder, anagrafica_data, config)
                    return 0


#
# @hydra.main(version_base=None, config_path="../configs", config_name="config")
# def main(config):
#     # Adjust paths in the config for cross-platform compatibility
#     data_source = format_path_for_os(config.settings.data_source)
#     data_output = format_path_for_os(config.settings.data_output)
#     background_images_folder = format_path_for_os(config.settings.background_images_folder)
#     output_directory_csv = format_path_for_os(config.settings.output_directory_csv)
#
#     # print(f"[+] Data source: {data_source}")
#     # print(f"[+] Data output: {data_output}")
#     # print(f"[+] Background images folder: {background_images_folder}")
#     # print(f"[+] Output directory csv: {output_directory_csv}")
#
#     # Retrieve directories or files based on the config settings
#     folders_in_data, message = get_directory_contents(data_source, 'folders')
#     if message:
#         print(message)
#         return
#
#     background_images_list = []
#     if config.settings.use_images:
#         background_images_list, image_message = get_directory_contents(background_images_folder, 'files')
#         if image_message:
#             print(image_message)
#             return
#
#     for folder in folders_in_data:
#         anagrafica_file = load_anagrafica(folder)
#         if not anagrafica_file:
#             print(f"[+] No Anagrafica file found in {folder}. Skipping...")
#             continue
#
#         anagrafica_data = read_anagrafica(anagrafica_file)
#         if not anagrafica_data:
#             print(f"Error reading or invalid anagrafica file format in {folder}. Skipping.")
#             continue
#
#         print(f"[+] Anagrafica data for {folder}: \n{anagrafica_data}")
#
#         task_file_list = sorted(
#             [f.path for f in os.scandir(folder) if f.is_file() and f.path.endswith(config.settings.file_extension)],
#             key=lambda x: len(x)
#         )
#
#         for task_number, task in enumerate(task_file_list):
#             if task_number == 0:
#                 if config.settings.use_images:
#                     preprocessing.process_images_files(folder, config.settings.images_extension, background_images_list)
#
#                 task_dataframe = preprocessing.load_data_from_csv(task)
#
#                 # Filter the dataframe by the type of points (onair / onpaper)
#                 # task_dataframe = preprocessing.points_type_filtering(task_dataframe,"onpaper")
#
#                 # Correct the coordinates system from Digitizer origin -> Image Standard origin
#                 task_dataframe = preprocessing.coordinates_manipulation(task_dataframe)
#
#                 print(f"[+] Task Preprocessed: \n {task_dataframe.head().to_string()} \n")
#
#     # # Check data directory
#     # folders_in_data = get_data_file(config)
#     # background_images_list = get_background_images_folder(config)
#     #
#     # # List of the empty Subjects' folder
#     # list_empty_folders = []
#     #
#     # # Iterate over the folders in data
#     # for folder in folders_in_data:
#     #     # Load the anagrafica file
#     #     anagrafica_file = load_anagrafica(folder)
#     #
#     #     if anagrafica_file is None:
#     #         # Skip the current folder
#     #         print(f"[+] No anagrafica file found. Skipping folder: {folder}")
#     #         list_empty_folders.append(folder)
#     #         continue
#     #     else:
#     #         # Read the anagrafica file
#     #         anagrafica_data = read_anagrafica(anagrafica_file)
#     #
#     #         print(f"[+] Anagrafica data: \n{anagrafica_data}")
#     #
#     #     # list of the image file in the subject folder
#     #     # images_folder = [f.path for f in os.scandir(folder) if f.is_dir()]
#     #
#     #     # Process the images in Images folder if necessary
#     #     # preprocessing.process_images_files(images_folder,config.settings.images_extension)
#     #
#     #     # Get all the .csv files (COMPLETE PATH) inside the folder
#     #     task_file_list = [f.path for f in os.scandir(folder) if
#     #                       f.is_file() and f.path.endswith(config.settings.file_extension)]
#     #
#     #     # Sort the file based on filename
#     #     task_file_list = sorted(task_file_list, key=lambda x: len(x))
#     #
#     #     # Iterate over the task file in the data folder and the background image
#     #     for task_number, (task, bck_image) in enumerate(zip(task_file_list, background_images_list)):
#     #         # Debug: computes only the "n" file in the folder
#     #         if task_number == 0:
#     #             # -------------------Preprocessing Section------------------- #
#     #             # Create the dataframe from the csv data
#     #             task_dataframe = preprocessing.load_data_from_csv(task)
#     #
#     #             # Filter the dataframe by the type of points (onair / onpaper)
#     #             # task_dataframe = preprocessing.points_type_filtering(task_dataframe,"onpaper")
#     #
#     #             # Correct the coordinates system from Digitizer origin -> Image Standard origin
#     #             task_dataframe = preprocessing.coordinates_manipulation(task_dataframe)
#     #
#     #             # Show image of the current task created from the csv
#     #             # preprocessing.create_image_from_data(task_dataframe["PointX"].to_numpy(),
#     #             #                                      task_dataframe["PointY"].to_numpy(),
#     #             #                                      task_dataframe["Pressure"].to_numpy(),
#     #             #                                      bck_image, task, config)
#     #
#     #             # Show image-video of the current task created from the csv
#     #             # preprocessing.create_gif_from_data(task_dataframe["PointX"].to_numpy(),
#     #             #                                      task_dataframe["PointY"].to_numpy(),
#     #             #                                      task_dataframe["Pressure"].to_numpy(),
#     #             #                                      bck_image, task, config)
#     #
#     #             # Print the csv data of the current task
#     #             # print(f"[+] Task Preprocessed: \n {task_dataframe.head().to_string()} \n")
#     #
#     #             # -------------------Library Conversion Section------------------- #
#     #             # Manipulate the dataframe to be ready for HandwritingSample library
#     #             task_dataframe = convert_to_HandwritingSample_library(task_dataframe)
#
#     # print(f"[+] Data after Transformation for HandwritingSample Library : \n{task_dataframe}")
#
#     # Debug: print the data after the transformation
#     # print(f"[+] Data after Transformation for HandwritingSample Library : \n{task_dataframe}")
#
#     # Get the strokes using the HandwritingSample library
#     # stroke_list = stroke_segmentation(task_dataframe)
#
#     # -------------------Feature Extraction Section------------------- #
#     # Stroke Approach feature extraction
#     # stroke_approach_dataframe = stroke_approach_feature_extraction(stroke_list, task_dataframe)
#
#     # print(f"[+] Stroke Approach Features: \n{stroke_approach_dataframe.to_string()}")
#
#     # Save the features extracted from the current task
#     # save_data_to_csv(stroke_approach_dataframe, task_number + 1, folder, anagrafica_data, config)
#     return 0
#


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    main_class = MainClass(config)
    main_class.run()


if __name__ == '__main__':
    main()
