import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from functools import wraps
from time import time
from datetime import datetime

# GLOBAL VARIABLE
# Wacom Device Clock (Hz)
CLOCK = 200

# Acquired Image dimensions
WIDTH_ACQUIRED = 1280
HEIGHT_ACQUIRED = 720

# WACOM ONE DIGITIZER VALUES
X_DIGITIZER = 29434
Y_DIGITIZER = 16556

# DESIRED OUTPUT IMAGE RESOLUTION
# WIDTH_IMAGE = 1600
# HEIGHT_IMAGE = 900

WIDTH_IMAGE = 1920
HEIGHT_IMAGE = 1080


def timing(f):
    """
    @timing = Decorator
    The `timing` function takes a function as an argument and returns a function that will print the
    time it takes to run the function

    :param f: the function to be timed
    :return: The function wrap is being returned.
    """

    @wraps(f)
    def wrap(*args, **kwargs):
        start_time = time()
        result = f(*args, **kwargs)
        elapsed_time = time()
        print(f'Function {f.__name__} took {elapsed_time - start_time:2.4f} seconds')
        return result

    return wrap


def calculate_age(date_string):
    """
    The `calculate_age` function takes a date string as an argument and returns the age of the person
    :param date_string: the date string to be converted
    :return: the age of the person
    """
    # Convert the date string to a datetime object
    date_format = "%d/%m/%Y"
    birth_date = datetime.strptime(date_string, date_format)

    # Get the current date
    current_date = datetime.now()

    # Calculate the difference between the current date and the birthdate
    age = current_date.year - birth_date.year

    # Adjust the age if the birthday hasn't happened yet this year
    if (
        birth_date.month > current_date.month
        or (birth_date.month == current_date.month and birth_date.day > current_date.day)
    ):
        age -= 1

    return age


def get_images_in_folder(images_folder_path: str, image_extension=".png") -> list:
    # Get all the images in the folder
    images_file_list = [f.path for f in os.scandir(images_folder_path) if
                        f.is_file() and f.path.endswith(image_extension)]

    # Sort by filename
    images_file_list = sorted(images_file_list, key=lambda x: len(x))

    return images_file_list


def process_images_files(images_folder_path: str, image_extension: str) -> None:
    """
    It takes a folder path and an image extension as input, and then it crops and resizes all the images
    in the folder
    
    :param images_folder_path: The path to the folder containing the images you want to process
    :type images_folder_path: str
    :param image_extension: The extension of the image files you want to process
    :type image_extension: str
    """

    # Get alle the images in the folder
    images_file_list = [f.path for f in os.scandir(images_folder_path) if f.is_file() and
                        f.path.endswith(image_extension)]

    # Sort by filename
    images_file_list = sorted(images_file_list, key=lambda x: len(x))

    # Loop through the images and crop and resize them if they don't match the size 1920x1080
    for file in images_file_list:
        crop_and_resize_image(file, file)


def crop_and_resize_image(source_path: str, destination_path: str) -> None:
    """
    It reads an image from the specified source path, crops it to the specified coordinates, resizes it
    to the specified dimensions, and saves it to the specified destination path
    
    :param source_path: The path to the image you want to crop and resize
    :type source_path: str
    :param destination_path: The path to the destination image
    :type destination_path: str
    """
    try:
        # Read the image
        img = cv2.imread(source_path)

        # Get image dimensions
        height, width, channels = img.shape

        # print(f"Width: {width} \t Height: {height} \n")

        if height != HEIGHT_IMAGE and width != WIDTH_IMAGE:
            # Crop the image to specified coordinates
            cropped = img[0:HEIGHT_ACQUIRED, 0:WIDTH_ACQUIRED]

            # Resize the cropped image to specified dimensions
            resized = cv2.resize(cropped, (WIDTH_IMAGE, HEIGHT_IMAGE))

            # Save the resized image
            cv2.imwrite(destination_path, resized)
        else:
            pass
    except FileNotFoundError:
        print(f"File {source_path} NOT found!")


def load_data_from_csv(file_csv: str) -> pd.DataFrame:
    """
    It reads the csv file line by line, separates the header from the rows, splits the header and rows
    by comma, removes the spaces, filters the desired values, creates a dataframe from the header and
    rows, pops the sequence, timestamp and PenId columns, and re-orders the dataframe columns
    
    :param file_csv: str 
    :type file_csv: str
    :return: Nothing.
    """
    try:
        # Read line by line the .csv file
        with open(file_csv) as file_csv:
            content = file_csv.readlines()

        # Separate Header and Rows
        header = content[:1]
        rows = content[1:]

        # Split the header and remove the spaces
        header = [x.strip() for xs in header for x in xs.split(',')]

        # Remove unused features
        header = header[0:5] + header[-7:]

        # Split the rows content by comma and remove spaces 
        rows = [row.replace(" ", "").strip().split(',', -1) for row in rows]

        # Filter only the desired values
        rows = [row[0:5] + row[-7:] for row in rows]

        # Create dataframe from header and rows
        task_data = pd.DataFrame(rows, columns=header)

        # Pop Sequence, Timestamp and PenId
        header.pop(header.index('Sequence'))
        header.pop(header.index('Timestamp'))
        header.pop(header.index('PenId'))

        # Re-order dataframe columns 
        # df = df[header + ['Timestamp']]
        task_data = task_data[header]

        # Cast columns values
        task_data = task_data.astype({'PointX': 'int32', 'PointY': 'int32', 'Pressure': 'int32',
                                      'Rotation': 'int32', 'Azimuth': 'int32', 'Altitude': 'int32',
                                      'TiltX': 'int32', 'TiltY': 'int32'})

        # Insert time column at the end of the dataframe
        # task_data["Time (s)"] = task_data.index * (1/CLOCK)

        # Insert Time column at the beginning
        task_data.insert(0, 'Time', task_data.index)

        task_data['Time'] = task_data['Time'] * (1 / CLOCK)

        return task_data

    except FileNotFoundError:
        print("File not found")


def points_type_filtering(df: pd.DataFrame, points_type: str = "onpaper") -> pd.DataFrame:
    """
    It takes a dataframe and a string as input, and returns a dataframe filtered by pressure
    
    :param df: the dataframe you want to filter
    :type df: pd.DataFrame
    :param points_type: str = "onpaper", defaults to onpaper
    :type points_type: str (optional)
    :return: A dataframe
    """

    if points_type == "onpaper":
        df = df.loc[df["Pressure"] != 0].reset_index(drop=True)
    elif points_type == "onair":
        df = df.loc[df["Pressure"] == 0].reset_index(drop=True)
    else:
        pass

    return df


def coordinates_manipulation(data: pd.DataFrame) -> pd.DataFrame:
    """
    It takes a dataframe of points, corrects the origin, filters the points, scales the points, and
    transforms the points to the image origin -> (0,0) to top-left.

    :param data: the dataframe containing the data
    :type data: pd.DataFrame
    :return: The x, y, and pressure arrays.
    """

    # Correct the mismatched origin between Digitizer and Screen
    # For plotting -> PointY must be uncommented
    # For Image -> PointY must be commented
    data["PointX"] = data.PointX.apply(lambda x_point: X_DIGITIZER - x_point)
    # data["PointY"] = data.PointY.apply(lambda y_point: Y_DIGITIZER - y_point)

    # Filter Points from dataframe
    x_coordinates_array = data["PointX"].to_numpy()
    y_coordinates_array = data["PointY"].to_numpy()

    mapped_x_values = map_coordinates(x_coordinates_array, 0, X_DIGITIZER, 0, WIDTH_IMAGE)
    mapped_y_values = map_coordinates(y_coordinates_array, 0, Y_DIGITIZER, 0, HEIGHT_IMAGE)

    data["PointX"] = mapped_x_values.astype(int)
    data["PointY"] = mapped_y_values.astype(int)

    # # Scale the arrays (Point_x : X_digitizer = Point_x_image : WIDTH_des) -> Point_x_image
    x_coordinates_array = (x_coordinates_array * WIDTH_IMAGE) / X_DIGITIZER
    y_coordinates_array = (y_coordinates_array * HEIGHT_IMAGE) / Y_DIGITIZER

    # Coordinates Transformation from Wacom Origin (TOP-RIGHT) to IMAGE Standard Coordinates Origin (TOP_LEFT)
    x_coordinates_array = WIDTH_IMAGE - (x_coordinates_array * 2)
    y_coordinates_array = HEIGHT_IMAGE - y_coordinates_array

    # Assign new x,y values
    data["PointX"] = x_coordinates_array.astype(int)
    data["PointY"] = y_coordinates_array.astype(int)

    return data


def map_coordinates(values, old_min, old_max, new_min, new_max):
    """ Function to mapping values from one range to another"""
    return (values - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    If you want to resize an image to a specific width, then pass the width as the first argument and
    the height as the second argument. If you want to resize an image to a specific height, then pass
    the height as the first argument and the width as the second argument
    
    :param image: The image we want to resize
    :param width: The width of the resized image
    :param height: The height we would like the image to be resized to
    :param inter: The interpolation method. By default, it is cv2.INTER_AREA for shrinking and
    cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR for zooming
    :return: The image is being resized to the dimensions of the width and height.
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def create_image_from_data(x: np.array, y: np.array, pressure: np.array, background_image: str, file_path: str,
                           config) -> None:
    """
    It takes 3 arrays and a string as input, and returns an image of the drawing.

    :param x: The x coordinates of the points
    :type x: np.array
    :param y: The y coordinates of the points
    :type y: np.array
    :param pressure: The pressure of the points
    :type pressure: np.array
    :param background_image: The path of the background image
    :type background_image: str
    :param file_path: The path of the file
    :type file_path: str
    :param config: The configuration file
    :type config: dict
    :return: None
    """

    # Settings for onpaper
    thickness = 2
    color = [0, 0, 0]

    # Settings for on air
    thickness_onair = 1
    color_onair = [0, 255, 0]

    # Read Background Image
    image = cv2.imread(background_image)

    # Merge 2 array 
    points = np.column_stack((x, y))

    # Loop through all the points for drawing on the canvas
    for i in range(1, len(points)):
        start = tuple(points[i - 1])
        end = tuple(points[i])

        # differentiate between onair and onpaper points
        if pressure[i] != 0:
            cv2.line(image, start, end, color, thickness)
        else:
            cv2.line(image, start, end, color_onair, thickness_onair)

    # Show the draw process in the Window
    cv2.imshow(file_path, image)

    cv2.waitKey(0)

    # Save the image
    saving_image(image, file_path, config)

    return None


def create_gif_from_data(x: np.array, y: np.array, pressure: np.array, background_image: str, file_path: str,
                         config) -> None:
    """
    > It takes in 3 arrays, and creates a video-like from them
    > Showing how the tasks have been built
    
    :param x: The x coordinates of the points
    :type x: np.array
    :param y: The y coordinates of the points
    :type y: np.array
    :param pressure: The pressure of the points
    :type pressure: np.array
    :param background_image: The path of the background image
    :type background_image: str
    :param file_path: The path of the file
    :type file_path: str
    :param config: The configuration file
    :type config: dict
    :return: None
    """

    # Settings for onpaper
    thickness = 2
    color = [0, 0, 0]

    # Settings for on air
    thickness_onair = 1
    color_onair = [0, 255, 0]

    # Read Background Image
    image = cv2.imread(background_image)

    # Merge 2 array 
    points = np.column_stack((x, y))

    # Loop through all the points for drawing on the canvas
    for i in range(1, len(points)):
        start = tuple(points[i - 1])
        end = tuple(points[i])

        # differentiate between onair and onpaper points
        if pressure[i] != 0:
            cv2.line(image, start, end, color, thickness)
        else:
            cv2.line(image, start, end, color_onair, thickness_onair)

            # Show the draw process in the Window
        cv2.imshow(file_path, image)

        # Pause the draw once finished
        if i == len(points) - 1:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)

    # Save the image
    saving_image(image, file_path, config)

    return None


def saving_image(image, original_filename: str, config) -> None:
    """
    It takes an image and a filename as input, and saves the image in the output folder with the same
    name as the original file
    
    :param image: the image to be saved
    :type image: np.array
    :param original_filename: The full path to the original image
    :type original_filename: str
    :param config: The configuration file
    :type config: dict
    :return: None
    """

    # Check if exists OUTPUT_FOLDER
    if not os.path.exists(config.settings.data_output):
        os.makedirs(config.settings.data_output)

    # Extract the filename from the complete path
    basename = os.path.splitext(os.path.basename(original_filename))[0]

    # obtain Task_#
    basename = basename.split('_')[0]

    # Retrieve the original folder name of the task
    original_subfolder_name = os.path.split(os.path.dirname(original_filename))[1]

    # Create if not exists the subfolder of the tasks
    sub_dir_output = os.path.join(config.settings.data_output, original_subfolder_name)

    if not os.path.exists(sub_dir_output):
        os.makedirs(sub_dir_output)

    # create the filename based on the image format desired
    filename = basename + config.settings.images_extension

    # Join the strings to obtain the complete path to save the image
    complete_path = os.path.join(sub_dir_output, filename)

    cv2.imwrite(complete_path, image)

    return None



