import os
from preprocessing import calculate_age, get_images_in_folder, process_images_files, load_data_from_csv, \
    coordinates_manipulation


class DataProcessor:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def format_path_for_os(path):
        return path.replace('\\', '/') if os.name == 'posix' else path

    @staticmethod
    def get_directory_contents(folder, content_type='folders'):
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            return [], f"Directory created at {folder}. Please insert the relevant contents."

        if not os.listdir(folder):
            return [], "Empty directory."

        if content_type == 'folders':
            return [f.path for f in os.scandir(folder) if f.is_dir()], None
        elif content_type == 'files':
            return get_images_in_folder(folder), None
        else:
            raise ValueError("Invalid content_type. Choose 'folders' or 'files'.")

    @staticmethod
    def load_anagrafica(folder):
        anagrafica_files = [f.path for f in os.scandir(folder) if f.is_file() and "Anagrafica" in f.name]
        return anagrafica_files[0] if anagrafica_files else None

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
    def process_images(folder, images_extension, background_images_list):

        # print(f"[+] Processing images in {folder}")
        # print(f"[+] Background images: {background_images_list}")
        # print(f"[+] Images extension: {images_extension}")
        images_folder = os.path.join(folder, "images")

        return process_images_files(images_folder, images_extension)

    @staticmethod
    def load_and_process_csv(task_file):
        dataframe = load_data_from_csv(task_file)
        # Additional processing can be added here
        dataframe = coordinates_manipulation(dataframe)
        return dataframe
