import sys

sys.dont_write_bytecode = True

import hydra
import os
from tqdm import tqdm
from data_processor import DataProcessor
import pandas as pd
import numpy as np
import src.plotting_module as plotting_module
import preprocessing
from data_conversion_module import convert_to_HandwritingSample_library, stroke_segmentation
from feature_extraction_strokes import stroke_approach_feature_extraction
from feature_extraction_statistic import statistical_feature_extraction


class MainClass:
    def __init__(self, config):
        self.config = config
        self.processor = DataProcessor(config)
        self.verbose = config.settings.verbose
        self.feature_extraction = config.settings.feature_extraction
        self.plot = config.settings.plotting
        self.task_list = config.settings.task_list

    def run(self):
        data_source = self.processor.format_path_for_os(self.config.settings.data_source)
        background_images_folder = self.processor.format_path_for_os(self.config.settings.background_images_folder)
        output_directory_csv = self.processor.format_path_for_os(self.config.settings.output_directory_csv)

        folders_in_data, message = self.processor.get_directory_contents(data_source, 'folders')
        if message:
            print(message)
            return 0

        background_images_list = []
        if self.config.settings.use_images:
            background_images_list, image_message = self.processor.get_directory_contents(background_images_folder,
                                                                                          'files')
            if image_message:
                print(image_message)
                return 0

        # Initialize the dataframe for the features
        subject_dataframe = pd.DataFrame()

        # Loop over Subject's folders
        for num_folder, folder in enumerate(tqdm(folders_in_data, desc="Processing Subject Folder:"), start=1):
            # Get the base path of the folder
            base_path = os.path.basename(os.path.normpath(folder))

            # Get the subject number from the folder name
            try:
                subject_number = int(base_path.split("_")[2])
            except ValueError:
                print(f"Invalid folder format: {base_path}")
                return 0

            # Load the anagrafica file
            anagrafica_file = self.processor.load_anagrafica(folder)
            if not anagrafica_file:
                print(f"[+] No anagrafica file found in {folder}. Skipping.")

                # Add the subject number to the dataframe with NaN values
                subject_dataframe = pd.concat([subject_dataframe,
                                               pd.DataFrame({"Id": subject_number, "Task": 0}, index=[0])],
                                              ignore_index=True)
                continue

            # Read the anagrafica file
            anagrafica_data = self.processor.read_anagrafica(anagrafica_file)
            if not anagrafica_data:
                print(f"Error reading or invalid anagrafica file format in {folder}. Skipping.")
                continue

            task_file_list = sorted(
                [f.path for f in os.scandir(folder) if
                 f.is_file() and f.path.endswith(self.config.settings.file_extension)],
                key=lambda x: len(x))

            if self.verbose:
                print(f"[+] Anagrafica data for SUBJECT {subject_number}: \n{anagrafica_data}")
                print(f"[+] Task file list: {task_file_list}")

            # Loop over the tasks in the folder of the subject
            for task_number, task in enumerate(task_file_list):
                # Debug: computes only the files in the task_list
                if task_number + 1 in self.task_list:
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

                    if self.verbose:
                        print(f"[+] Task {task_number + 1} dataframe: \n{task_dataframe.head(10)}")

                    if self.plot:
                        if self.verbose:
                            print(f"[+] Plotting 3D for task {task_number + 1} of subject {subject_number}")

                        plotting_module.plot_3d()

                    # region Feature Extraction
                    if self.feature_extraction:
                        # -------------------Library Conversion Section------------------- #
                        # Manipulate the dataframe to be ready for HandwritingSample library
                        task_dataframe = convert_to_HandwritingSample_library(task_dataframe)

                        # Debug: print the data after the transformation
                        if self.verbose:
                            print(f"[+] Data after Transformation for HandwritingSample Library : \n"
                                  f"{task_dataframe.head(10).to_string()} \n")

                        # Get the strokes using the HandwritingSample library
                        # stroke_list = stroke_segmentation(task_dataframe)

                        # -------------------Feature Extraction Section------------------- #
                        # Get the features using the HandwritingSample library from the task
                        handwriting_feature_dict = statistical_feature_extraction(task_dataframe,
                                                                                  self.config.settings.in_air)

                        task_dict_complete = {"Id": subject_number, **handwriting_feature_dict,
                                              **anagrafica_data, "Task": task_number + 1}

                        # Concatenate the features extracted from current task to the dataframe with incremented index
                        subject_dataframe = pd.concat([subject_dataframe,
                                                       pd.DataFrame(task_dict_complete, index=[0])],
                                                      ignore_index=True)

                        # Stroke Approach feature extraction
                        # stroke_approach_dataframe = stroke_approach_feature_extraction(stroke_list, task_dataframe)

                        # print(f"[+] Stroke Approach Features: \n{stroke_approach_dataframe.to_string()}")

                        # Save the features extracted from the current task
                        # save_data_to_csv(stroke_approach_dataframe, task_number + 1, folder, anagrafica_data, config)
                    # endregion

        if self.verbose and subject_dataframe.shape[0] > 0:
            print(f"[+] Subject dataframe: \n{subject_dataframe.to_string()}")
            print(f"[+] Subject dataframe shape: {subject_dataframe.shape}")

        # Check if subject_dataframe is empty
        if self.feature_extraction:
            # Order the dataframe by Task
            subject_dataframe = subject_dataframe.sort_values(by=['Id', 'Task'])

            # Get the unique Task values
            unique_task_values = subject_dataframe['Task'].unique()

            # Loop over the unique Task values and extract the rows with the same Task value
            for task in unique_task_values:
                task_dataframe = subject_dataframe.loc[subject_dataframe['Task'] == task].reset_index(drop=True)

                # Save the dataframe to csv into output_directory_csv
                task_dataframe.to_csv(os.path.join(output_directory_csv, f"Task_{task}.csv"), index=False)

                if self.verbose:
                    print(f"[+] Task {task}: \n{task_dataframe.to_string()}")

        return 0


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    main_class = MainClass(config)
    main_class.run()


if __name__ == '__main__':
    main()
