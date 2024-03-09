import os
from pathlib import Path

import pandas as pd

from preprocessing import calculate_age, get_images_in_folder, process_images_files, load_data_from_csv, \
    coordinates_manipulation


class DataProcessor:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def format_path_for_os(path):
        return Path(path).as_posix()

    @staticmethod
    def get_directory_contents(folder, content_type='folders'):
        folder_path = Path(folder)
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            return [], f"Directory created at {folder}. Please insert the relevant contents."

        if not any(folder_path.iterdir()):
            return [], "Empty directory."

        if content_type == 'folders':
            return [f.as_posix() for f in folder_path.iterdir() if f.is_dir()], None
        elif content_type == 'files':
            return get_images_in_folder(folder_path), None
        else:
            raise ValueError("Invalid content_type. Choose 'folders' or 'files'.")

    @staticmethod
    def load_anagrafica(folder):
        folder_path = Path(folder)
        anagrafica_files = [f for f in folder_path.iterdir() if f.is_file() and "Anagrafica" in f.name]
        return anagrafica_files[0].as_posix() if anagrafica_files else None

    @staticmethod
    def read_anagrafica(anagrafica_file):
        try:
            with open(anagrafica_file, "r") as file:
                data = file.readlines()

            gender = data[3].split(":")[1].strip()
            dob = data[4].split(":")[1].strip()
            dominant_hand = data[5].split(":")[1].strip()
            age = calculate_age(dob)

            return {"Gender": gender, "Age": age, "Dominant_Hand": dominant_hand}
        except (OSError, IndexError):
            print(f"Error while reading or processing: {anagrafica_file}")
            return None

    @staticmethod
    def process_images(folder: str, images_extension: str, verbose: bool = False):
        if verbose:
            print(f"\n[+] Processing images in {folder}")
            print(f"[+] Images extension: {images_extension}")

        images_folder = Path(folder) / "images"
        return process_images_files(images_folder_path=images_folder, image_extension=images_extension)

    @staticmethod
    def load_and_process_csv(task_file: str) -> pd.DataFrame:
        """
        Load the csv file and process the data
        :param task_file: str, path to the csv file
        :return: pd.DataFrame, processed data

        """
        task_df = load_data_from_csv(file_csv=task_file)
        # Correct the coordinates system from Digitizer origin -> Image Standard origin
        task_df = coordinates_manipulation(data=task_df)
        return task_df
