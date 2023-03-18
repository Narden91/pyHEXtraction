import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# GLOBAL VARIABLE
# Wacom Device Clock (Hz)
CLOCK = 200

# WACOM ONE DIGITIZER VALUES
X_DIGITIZER = 29434
Y_DIGITIZER = 16556 

# DESIRED OUTPUT IMAGE RESOLUTION
WIDTH_IMAGE = 1600
HEIGHT_IMAGE = 900

# WIDTH_IMAGE = 1920
# HEIGHT_IMAGE = 1080


def process_images_files(images_folder_path: str, image_extension: str):
    """
    It takes a folder path and an image extension as input, and then it crops and resizes all the images
    in the folder
    
    :param images_folder_path: The path to the folder containing the images you want to process
    :type images_folder_path: str
    :param image_extension: The extension of the image files you want to process
    :type image_extension: str
    """
    
    for image in images_folder_path:
            # Get alle the images in the folder
            images_file_list = [f.path for f in os.scandir(image) if f.is_file() and f.path.endswith(image_extension)]
            
            # Sort by filename
            images_file_list = sorted(images_file_list, key = lambda x: len(x))
                        
            # Loop through the images and crop and resize them if they don't match the size 1920x1080
            for file in images_file_list:
                
                crop_and_resize_image(file,file)


def crop_and_resize_image(source_path:str, destination_path:str) -> None:
    """
    It reads an image from the specified source path, crops it to the specified coordinates, resizes it
    to the specified dimensions, and saves it to the specified destination path
    
    :param source_path: The path to the image you want to crop and resize
    :type source_path: str
    :param destination_path: The path to the destination image
    :type destination_path: str
    """
    # Read the image
    img = cv2.imread(source_path)
    
    # Get image dimensions
    height, width, channels = img.shape
    
    # print(f"Width: {width} \t Height: {height} \n")

    if height != 1080 and width != 1920:   
        # Crop the image to specified coordinates
        cropped = img[0:720, 0:1280]
        
        # Resize the cropped image to specified dimensions
        resized = cv2.resize(cropped, (1920, 1080))

        # Save the resized image
        cv2.imwrite(destination_path, resized)
    else:
        pass
        
   
def load_data_from_csv(file_csv: str) -> pd.DataFrame:
    """
    It reads the csv file line by line, separates the header from the rows, splits the header and rows
    by comma, removes the spaces, filters the desired values, creates a dataframe from the header and
    rows, pops the sequence, timestamp and PenId columns, and re-orders the dataframe columns
    
    :param file_csv: str = 'C:/Users/user/Desktop/test.csv'
    :type file_csv: str
    :return: Nothing.
    """
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
    task_data = pd.DataFrame(rows,columns=header)
    
    # Pop Sequence, Timestamp and PenId
    header.pop(header.index('Sequence')) 
    header.pop(header.index('Timestamp')) 
    header.pop(header.index('PenId')) 
    
    # Re-order dataframe columns 
    # df = df[header + ['Timestamp']]
    task_data = task_data[header]
    
    # Cast columns values
    task_data = task_data.astype({'PointX': 'int32', 'PointY': 'int32','Pressure': 'int32',
                    'Rotation': 'int32', 'Azimuth': 'int32', 'Altitude': 'int32',
                    'TiltX': 'int32', 'TiltY': 'int32'})
    
    # Insert time column at the end of the dataframe
    # task_data["Time (msec)"] = task_data.index * (1/CLOCK)
    
    # Insert Time column at the beginning
    task_data.insert(0, 'Time (msec)', task_data.index)
    
    task_data['Time (msec)'] = task_data['Time (msec)'] * (1/CLOCK) 

    return task_data


def points_type_filtering(df:pd.DataFrame, points_type:str = "onpaper") -> pd.DataFrame:
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


def coordinates_manipulation(data:pd.DataFrame) :
    """
    It takes a dataframe of points, corrects the origin, filters the points, scales the points, and
    transforms the points to the image origin
    
    :param data: the dataframe containing the data
    :type data: pd.DataFrame
    :return: The x, y, and pressure arrays.
    """
    
    # Correct the mismatched origin between Digitizer and Screen
    data["PointX"] = data.PointX.apply(lambda x_point: X_DIGITIZER - x_point)
    data["PointY"] = data.PointY.apply(lambda y_point: Y_DIGITIZER - y_point)
    
    # Filter Points from dataframe
    x_coordinates_array = data["PointX"].to_numpy()
    y_coordinates_array = data["PointY"].to_numpy()
        
    # Scale the arrays (Point_x : X_digitizer = Point_x_image : WIDTH_des) -> Point_x_image   
    x_coordinates_array = (x_coordinates_array * WIDTH_IMAGE) / X_DIGITIZER
    y_coordinates_array = (y_coordinates_array * HEIGHT_IMAGE) / Y_DIGITIZER
        
    # Coordinates Transformation from Wacom Origin (TOP-RIGHT) to IMAGE Standard Coordinates Origin (TOP_LEFT)
    x_coordinates_array = WIDTH_IMAGE - (x_coordinates_array * 2)
    y_coordinates_array = HEIGHT_IMAGE - y_coordinates_array
    
    # Assign new x,y values
    data["PointX"] = x_coordinates_array.astype(int)
    data["PointY"] = y_coordinates_array.astype(int)
    
    return data
    
    
def create_array_plot(x:np.array, y:np.array) -> None:
    """
    It takes two arrays, x and y, and plots them on a graph
    
    :param x: the x-axis values
    :type x: np.array
    :param y: np.array = The y-axis values
    :type y: np.array
    :return: None
    """
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    
    plt.title("Task Plot")
    plt.plot(x, y, color="black")

    plt.xlim([0, WIDTH_IMAGE])
    plt.ylim([0, HEIGHT_IMAGE])
    plt.show()
    
    return None
        
    
def create_image_from_data(x:np.array, y:np.array, pressure:np.array) -> None:
    """
    > It takes in 3 arrays, and creates an image from them
    
    :param x: x-coordinates of the points
    :type x: np.array
    :param y: the y-coordinates of the points
    :type y: np.array
    :param pressure: the pressure of the pen on the tablet
    :type pressure: np.array
    """
    
    # Settings for onpaper
    thickness = 2
    color = [0,0,0]
    
    # Settings for on air
    thickness_onair = 1
    color_onair = [0,255,0]
    
    # Create Image Matrix
    image = np.zeros((HEIGHT_IMAGE, WIDTH_IMAGE, 3), np.uint8)
    
    # Fill the Image with white color
    image.fill(255)
    
    # Merge 2 array 
    points = np.column_stack((x, y))
    
    # Loop through all the points for drawing on the canvas
    for i in range(1, len(points)):
        start = tuple(points[i - 1])
        end = tuple(points[i])
        
        if pressure[i] != 0:
            cv2.line(image, start, end, color, thickness)
        else:
            cv2.line(image, start, end, color_onair, thickness_onair)  
   
    # Show the image in a new Window
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
    # Save locally the image created
    filename = 'savedImage.jpg'
    
    cv2.imwrite(filename, image)
    
    return None


def compute_speed_and_acceleration(x:np.array, y:np.array):
    """
    It computes the speed and acceleration of the traits given its x and y coordinates
    
    :param x: the x-coordinates of the points
    :type x: np.array
    :param y: the y-coordinates of the trajectory
    :type y: np.array
    :return: the speed and acceleration of the object.
    """
    
    #  Calculate the n-th discrete difference along the given axis
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(np.arange(len(x))) 
    
    # Velocity
    velocity = np.sqrt(dx**2 + dy**2) / dt
    
    
    dv = np.diff(velocity)
    
    # Acceleration
    acceleration = dv / dt[:-1]
    acceleration = np.append(acceleration, 0)
    acceleration = np.append(acceleration, 0)
    velocity = np.append(velocity, 0)
        
    return velocity , acceleration