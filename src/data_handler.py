import os
import pandas as pd
import preprocessing
import sys

sys.dont_write_bytecode = True


def get_data_folders(folder):
    """
    Check the data directory and retrieve a list of subdirectories.

    Args:
    folder (str): Path to the data directory.

    Returns:
    list: List of paths to subdirectories within the data directory.
    """
    # Ensure the folder exists, create if it doesn't
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return [], "Data directory created. Please insert folders and files."

    # Check if the directory is empty
    if not os.listdir(folder):
        return [], "Empty data directory."

    # Get all subdirectories in the folder
    folders_in_data = [f.path for f in os.scandir(folder) if f.is_dir()]
    return folders_in_data, None


def get_background_images_folder(folder):
    """
    Check the background images directory and retrieve the background images
    :param folder: background images directory
    :return: background_images_list: list of the background images
    """
    # Check background images directory
    try:
        # If directory has not yet been created
        os.makedirs(folder)

        sys.exit("Empty background images folder")
    except OSError:
        if not os.listdir(folder):
            sys.exit("Empty background images folder")
        else:
            # Get the background_images
            background_images_list = preprocessing.get_images_in_folder(folder)

    return background_images_list


def get_directory_contents(folder, content_type='folders'):
    """
    Check a directory and retrieve its contents (subdirectories or files).

    Args:
    folder (str): Path to the directory.
    content_type (str): Type of content to retrieve ('folders' or 'files').

    Returns:
    list: List of paths to the requested contents within the directory.
    str: Message indicating the status or any issues with the directory.
    """
    # Ensure the folder exists, create if it doesn't
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return [], f"Directory created at {folder}. Please insert the relevant contents."

    # Check if the directory is empty
    if not os.listdir(folder):
        return [], "Empty directory."

    # Get contents based on the specified type
    if content_type == 'folders':
        contents = [f.path for f in os.scandir(folder) if f.is_dir()]
    elif content_type == 'files':
        contents = preprocessing.get_images_in_folder(folder)
    else:
        raise ValueError("Invalid content_type. Choose 'folders' or 'files'.")

    return contents, None


def load_anagrafica(folder):
    """
    Load the anagrafica file from the folder.
    :param folder: folder containing the anagrafica file
    :return: the path of the anagrafica file or None if not found
    """
    anagrafica_files = [f.path for f in os.scandir(folder) if f.is_file() and "Anagrafica" in f.name]
    return anagrafica_files[0] if anagrafica_files else None


def read_anagrafica(anagrafica_file):
    """
    Read the anagrafica text file and extract relevant data.
    :param anagrafica_file: path of the anagrafica file
    :return: dictionary containing the anagrafica data {Gender, Age, Dominant_Hand} or None if an error occurs
    """
    try:
        with open(anagrafica_file, "r") as file:
            data = file.readlines()

        gender = data[3].split(":")[1].strip()
        dob = data[4].split(":")[1].strip()
        dominant_hand = data[5].split(":")[1].strip()
        age = preprocessing.calculate_age(dob)

        return {"Gender": gender, "Age": age, "Dominant_Hand": dominant_hand}
    except (OSError, IndexError):
        print(f"Error while reading or processing: {anagrafica_file}")
        return None


def save_data_to_csv(dataframe, task_number, folder, anagrafica_dict, config):
    """
    Save the dataframe to csv
    :param dataframe: dataframe to save
    :param task_number: number of the task
    :param folder: folder in which save the csv
    :param anagrafica_dict: dictionary containing the anagrafica data
    :param config: config file
    :return: None
    """
    # Create the filename for the csv
    filename_csv = "Task_" + str(task_number) + ".csv"

    # print(f"[+] Saving dataframe to csv: {filename_csv} \n")

    # Get the last part of the path (the folder name)
    folder = os.path.split(folder)[1]

    # Get Subject number (CRC_SUBJECT_001 -> 1)
    subject_number = int(folder.split("_")[2])

    # Add subject information to the dataframe
    dataframe = add_personal_info_to_dataframe(dataframe, subject_number, anagrafica_dict, task_number)

    # Check if the directory exists
    try:
        # If the directory has not yet been created
        os.makedirs(config.settings.output_directory_csv)
    except OSError:
        pass

    try:
        task_path = os.path.join(config.settings.output_directory_csv, folder)
        os.makedirs(task_path)
    except OSError:
        pass

    try:
        dataframe.to_csv(task_path + "/" + filename_csv, index=False)
    except FileNotFoundError:
        print("File not found")


def add_personal_info_to_dataframe(dataframe: pd.DataFrame, id_subject: int, anagrafica_dict: dict, task_number: int):
    """
    Add personal information to the dataframe
    :param dataframe: dataframe to which add the personal information
    :param id_subject: id of the subject
    :param anagrafica_dict: dictionary containing the anagrafica data
    :param task_number: task number
    :return: pd.DataFrame
    """

    # Add a column with the subject number at the beginning of the dataframe
    dataframe.insert(0, "Subject", id_subject)

    # Add columns with the subject info from the anagrafica at the end of the dataframe
    dataframe.insert(len(dataframe.columns), "Gender", anagrafica_dict["Gender"])
    dataframe.insert(len(dataframe.columns), "Age", anagrafica_dict["Age"])
    dataframe.insert(len(dataframe.columns), "Dominant_Hand", anagrafica_dict["Dominant_Hand"])

    # Add a column with the task number at the end of the dataframe
    dataframe.insert(len(dataframe.columns), "Task", task_number)

    return dataframe
