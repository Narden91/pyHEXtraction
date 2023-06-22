import os

import pandas as pd

import preprocessing
import sys

sys.dont_write_bytecode = True


def check_directories_and_load_data(config):
    # Check data directory
    try:
        # If a directory has not yet been created
        os.makedirs(config.settings.data_source)

        sys.exit("Empty directory data")
    except OSError:
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
    except OSError:
        if not os.listdir(config.settings.background_images_folder):
            sys.exit("Empty background images folder")
        else:
            # Get the background_images
            background_images_list = preprocessing.get_images_in_folder(config.settings.background_images_folder)

    return folders_in_data, background_images_list


def load_anagrafica(folder):
    """
    Load the anagrafica file from the folder
    :param folder: folder containing the anagrafica file
    :return: the path of the anagrafica file
    """

    anagrafica_list = [f.path for f in os.scandir(folder) if f.is_file() and "Anagrafica" in f.name]

    # Check if there is an anagrafica file
    if len(anagrafica_list) == 0:
        sys.exit("No anagrafica file found")
    else:
        return anagrafica_list[0]


def read_anagrafica(anagrafica_file):
    """
    Read the anagrafica text file row by row
    :param anagrafica_file: path of the anagrafica file
    :return anagrafica_data: dictionary containing the anagrafica data {Gender, Age, Dominant_Hand}
    """

    try:
        # Open the anagrafica file
        with open(anagrafica_file, "r") as f:
            # Read the file row by row
            anagrafica_data = f.readlines()

        # get the subject sex from the fourth row after the string Sesso:
        subject_gender = anagrafica_data[3].split(":")[1].strip()

        # get the subject date of birth from the fifth row after the string Data di nascita:
        subject_date_of_birth = anagrafica_data[4].split(":")[1].strip()

        # get the subject age given the date of birth by subtracting the current year
        subject_age = preprocessing.calculate_age(subject_date_of_birth)

        # get the subject dominant hand from the sixth row after the string Mano dominante:
        subject_dominant_hand = anagrafica_data[5].split(":")[1].strip()

        # Aggregate all the data in a dictionary
        anagrafica_data = {"Gender": subject_gender,
                           "Age": subject_age,
                           "Dominant_Hand": subject_dominant_hand}

        return anagrafica_data
    except OSError:
        raise Exception(f"Error while reading: {anagrafica_file} file")


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
