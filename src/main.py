import sys
from pathlib import Path
import hydra
import os
from tqdm import tqdm
from data_processor import DataProcessor
import pandas as pd
import src.plotting_module as plotting_module
import preprocessing
from data_conversion_module import convert_to_HandwritingSample_library_format, stroke_segmentation
from feature_extraction_strokes import stroke_approach_feature_extraction
from feature_extraction_statistic import statistical_feature_extraction

sys.dont_write_bytecode = True


class MainClass:
    def __init__(self, config):
        self.config = config
        self.processor = DataProcessor(config)
        self.verbose = config.settings.verbose
        self.show_images = config.settings.show_images
        self.show_gif = config.settings.show_gif
        self.elaborate_images = config.settings.create_online_images
        self.feature_extraction = config.settings.feature_extraction
        self.fs_approach = config.settings.fs_approach
        self.plot = config.settings.plotting
        self.task_list = config.settings.task_list
        self.score = config.settings.score_file
        self.overwrite_flag = config.settings.overwrite
        self.subject_task_to_plot = config.settings.subject_task_to_plot
        self.output_task_3d = config.settings.output_task_3d

    @staticmethod
    def get_subject_number(folder):
        try:
            return int(folder.name.split("_")[2])
        except ValueError as e:
            raise ValueError(f"Invalid folder format: {folder.name}") from e

    def load_and_process_anagrafica(self, folder):
        anagrafica_file = self.processor.load_anagrafica(folder)
        if anagrafica_file:
            return self.processor.read_anagrafica(anagrafica_file)
        else:
            if self.verbose:
                print(f"[+] No anagrafica file found in {folder}. Skipping.")
            return None

    def get_task_file_list(self, folder):
        return sorted(folder.glob(f'*{self.config.settings.file_extension}'), key=lambda x: len(x.name))

    def run(self):
        data_source = self.processor.format_path_for_os(self.config.settings.data_source)
        background_images_folder = self.processor.format_path_for_os(self.config.settings.background_images_folder)
        output_directory_csv = Path(self.processor.format_path_for_os(self.config.settings.output_directory_csv))
        plotting_output_directory = self.processor.format_path_for_os(self.config.settings.output_directory_plots)
        output_task_3d = self.processor.format_path_for_os(self.output_task_3d)

        output_directory_csv.mkdir(parents=True, exist_ok=True)

        # Read the score file
        score_df = pd.read_csv(self.score, delimiter=';').fillna(-1).astype(int)

        folders_in_data, message = self.processor.get_directory_contents(data_source, 'folders')
        if message:
            print(message)
            return 0

        if self.elaborate_images:
            bckg_images_list, image_message = self.processor.get_directory_contents(background_images_folder,
                                                                                    'files')
            if image_message:
                print(image_message)
                return 0
        else:
            bckg_images_list = [None] * len(self.task_list)

        # Initialize the dataframe for the features
        subject_dataframe = pd.DataFrame()
        subjects_missing_anagrafica = set()

        print(f"[+] Folders in data: {len(folders_in_data)}") if self.verbose else None

        # Loop over Subject's folders
        for num_folder, folder in enumerate(tqdm(folders_in_data, desc="Processing Subject Folder"), start=1):
            if num_folder < 2:
                folder = Path(folder)
                subject_number = self.get_subject_number(folder=folder)

                anagrafica_data = self.load_and_process_anagrafica(folder=folder)
                if not anagrafica_data:
                    subjects_missing_anagrafica.add(subject_number)
                    print(f"[-] Anagrafica data missing for SUBJECT {subject_number}. Skipping.")
                    continue

                task_file_list = self.get_task_file_list(folder=folder)

                if self.verbose:
                    print(f"[+] Anagrafica data for SUBJECT {subject_number}: \n {anagrafica_data}")
                    print(f"[+] Subject folder: {folder}")
                    print(f"[+] Task file list: {len(task_file_list)}")

                # if subjects_missing_anagrafica:
                #     print(f"[-] Subjects missing anagrafica data:{sorted(subjects_missing_anagrafica)}")

                # Loop over the tasks in the folder of the subject
                for task_number, task in enumerate(task_file_list):
                    # Debug: computes only the files in the task_list
                    if task_number + 1 in self.task_list:  # Modify CONFIG for all tasks
                        # -------------------Preprocessing Section------------------- #
                        task_df = self.processor.load_and_process_csv(task_file=task)

                        # Add the subject label from the score file for the current task
                        # task_dataframe['Label'] = \
                        #     score_df.loc[score_df['Id'] == subject_number, str(task_number + 1)].values[0]

                        sbj_task_label = score_df.loc[score_df['Id'] == subject_number, str(task_number + 1)].values[0]

                        # Filter the dataframe by the type of points (onair / onpaper)
                        # task_dataframe = preprocessing.points_type_filtering(task_dataframe,"onpaper")

                        # region Plotting
                        if self.plot:
                            if self.verbose:
                                print(f"[+] Plotting 3D for task {task_number + 1} of subject {subject_number}")

                            task_path_folder = Path(plotting_output_directory) / f"Task_{task_number + 1}"
                            task_path_folder.mkdir(parents=True, exist_ok=True)
                            task_filename = task_path_folder / f"Subject_{subject_number}.csv"

                            # Save the task_dataframe to csv if it does not exist or overwrite is enabled
                            if not task_filename.exists() or self.overwrite_flag:
                                task_df.to_csv(task_filename, index=False)

                            # Plot Task in 3D and 2D if the subject is in the specified list to plot
                            if subject_number in self.subject_task_to_plot:
                                plotting_module.plot_task(df=task_df,
                                                          subject_number=subject_number,
                                                          task_number=task_number + 1,
                                                          output_directory=output_task_3d)
                        # endregion

                        # region Image and GIF Creation
                        if self.elaborate_images:
                            self.processor.process_images(folder, self.config.settings.images_extension, self.verbose)

                            # Show image of the current task created from the csv
                            preprocessing.create_image_from_data(x=task_df["PointX"].to_numpy(),
                                                                 y=task_df["PointY"].to_numpy(),
                                                                 pressure=task_df["Pressure"].to_numpy(),
                                                                 background_image=bckg_images_list[task_number],
                                                                 file_path=task, show_image=self.show_images,
                                                                 config=self.config)

                            # Show image-video of the current task created from the csv
                            if self.show_gif:
                                preprocessing.create_gif_from_data(x=task_df["PointX"].to_numpy(),
                                                                   y=task_df["PointY"].to_numpy(),
                                                                   pressure=task_df["Pressure"].to_numpy(),
                                                                   background_image=bckg_images_list[task_number],
                                                                   file_path=task,
                                                                   config=self.config)
                        # endregion

                        # region Feature Extraction
                        if self.feature_extraction:
                            # -------------------Library Conversion Section------------------- #
                            # Manipulate the dataframe to be ready for HandwritingSample library
                            task_df = convert_to_HandwritingSample_library_format(task_df, self.verbose)

                            if self.verbose:
                                print(f"[+] Data after Transformation for HandwritingSample Library : \n"
                                      f"{task_df.head(10).to_string()} \n")

                            if self.fs_approach.lower() == "statistical":
                                # -------------------Statistical Feature Extraction Section------------------- #
                                if self.verbose:
                                    print(f"[+] Statistical Feature Extraction for task {task_number + 1}")
                                # Get the features using the HandwritingSample library from the task
                                handwriting_feature_dict = statistical_feature_extraction(task_dataframe=task_df,
                                                                                          verbose=self.verbose)

                                task_dict_complete = {"Id": subject_number, **handwriting_feature_dict,
                                                      **anagrafica_data, "Task": task_number + 1,
                                                      "Label": sbj_task_label}

                                # Remap the features to the dataframe
                                features_values = task_dict_complete['features'].flatten()
                                features_labels = task_dict_complete['labels']
                                features_dict = {label: value for label, value in zip(features_labels, features_values)}

                                task_dict_complete = {k: v for k, v in task_dict_complete.items()
                                                      if k not in ['features', 'labels']}
                                task_dict_complete.update(features_dict)

                                subject_dataframe = pd.concat([subject_dataframe, pd.DataFrame(task_dict_complete,
                                                                                               index=[0])],
                                                              ignore_index=True)

                            elif self.fs_approach.lower() == "stroke":
                                # -------------------Stroke Feature Extraction Section------------------- #
                                if self.verbose:
                                    print(f"[+] Stroke Feature Extraction for task {task_number + 1}")
                                # Get the strokes using the HandwritingSample library
                                stroke_list = stroke_segmentation(data_source=task_df, verbose=self.verbose)

                                subject_dataframe = stroke_approach_feature_extraction(strokes=stroke_list,
                                                                                       task_dataframe=task_df,
                                                                                       verbose=self.verbose)

                                subject_dataframe['Id'] = subject_number
                                subject_dataframe['Gender'] = anagrafica_data["Gender"]
                                subject_dataframe['Age'] = anagrafica_data["Age"]
                                subject_dataframe['Dominant_Hand'] = anagrafica_data["Dominant_Hand"]
                                subject_dataframe['Task'] = task_number + 1
                                subject_dataframe['Label'] = sbj_task_label

                                if self.verbose:
                                    print(f"[+] Stroke Approach Features: \n{subject_dataframe.to_string()}")
                        # endregion

            if self.verbose and subject_dataframe.shape[0] > 0:
                print(f"[+] Subject dataframe: \n{subject_dataframe.to_string()}")
                print(f"[+] Subject dataframe shape: {subject_dataframe.shape}")

            if self.feature_extraction:
                columns_to_move = ["Gender", "Age", "Dominant_Hand", "Label", "Task"]
                other_columns = [col for col in subject_dataframe.columns if col not in columns_to_move]
                subject_dataframe = subject_dataframe[other_columns + columns_to_move]

                subject_dataframe = subject_dataframe.sort_values(by=['Id', 'Task'])
                unique_task_values = subject_dataframe['Task'].unique()

                for task in unique_task_values:
                    task_df = subject_dataframe.loc[subject_dataframe['Task'] == task].reset_index(drop=True)

                    if self.fs_approach.lower() == "statistical":
                        task_path_folder = Path(output_directory_csv) / f"Task_{task}.csv"
                        task_df.to_csv(task_path_folder, index=False)
                    elif self.fs_approach.lower() == "stroke":
                        task_path_folder = Path(output_directory_csv) / f"Task_{task}_stroke.csv"
                        task_df.to_csv(task_path_folder, index=False)

                    if self.verbose:
                        print(f"[+] Task {task}: \n{task_df.to_string()}")
            else:
                print(f"[-] No feature extracted for Subject {subject_number}")
        return 0


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    main_class = MainClass(config)
    main_class.run()


if __name__ == '__main__':
    main()
