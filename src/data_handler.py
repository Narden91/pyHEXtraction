import os
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


def save_data_to_csv(dataframe, task_number, folder, filename, config):
    """
    Save the dataframe to csv
    :param dataframe: dataframe to save
    :param task_number: number of the task
    :param folder: folder in which save the csv
    :param filename: name of the csv file
    :param config: config file
    :return: None
    """

    # Get the last part of the file path (the filename)
    filename = os.path.split(filename)[1]

    # Obtain Task number from filename
    filename = filename.split("_")[0] + ".csv"

    # Get the last part of the path (the folder name)
    folder = os.path.split(folder)[1]

    # Get Subject number (CRC_SUBJECT_001 -> 1)
    subject_number = int(folder.split("_")[2])

    # Add a column with the subject number at the beginning of the dataframe
    dataframe.insert(0, "Subject", subject_number)

    # Add a column with the task number at the end of the dataframe
    dataframe.insert(len(dataframe.columns), "Task", task_number)

    task_path = ""

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
        print(f"[+] Dataframe saved: \n {dataframe.to_string()} \n")
        # Save the dataframe to csv
        dataframe.to_csv(task_path + "/" + filename , index=False)
    except OSError:
        raise Exception(f"Error while saving: {filename} file")
