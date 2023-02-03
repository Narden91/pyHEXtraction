import pandas as pd
import numpy as np
import cv2
import os


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
        
   
def dataframe_from_csv(file_csv: str) -> pd.DataFrame:
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
    df = pd.DataFrame(rows,columns=header)
    
    # Pop Sequence, Timestamp and PenId
    header.pop(header.index('Sequence')) 
    header.pop(header.index('Timestamp')) 
    header.pop(header.index('PenId')) 
    
    # Re-order dataframe columns 
    df = df[header + ['Timestamp']]
    
    df = df.astype({'PointX': 'int32', 'PointY': 'int32','Pressure': 'int32', 'Rotation': 'int32', 'Azimuth': 'int32', 'Altitude': 'int32', 'TiltX': 'int32', 'TiltY': 'int32'})
  
    return df


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


def process_and_create_arrays_points(data:pd.DataFrame) :
    """
    Function that takes a dataframe, filter the points Columns, create 2 array of points
    and returns two arrays without offset and scaled with their factors 
    
    :param array_x: the x-coordinates of the points
    :type array_x: np.array
    :param array_y: the y-coordinates of the points
    :type array_y: np.array
    :return: the scaled x and y arrays.
    """
    
    # Filter Points from dataframe
    array_x = data["PointX"].to_numpy()
    
    array_y = data["PointY"].to_numpy()
    
    # Compute the offset
    array_x = array_x - 0
    
    array_y = array_y - 0
    
    # Scale the arrays
    array_x = array_x * 1920/29400
    
    array_y = array_y * 1080/16600
    
    return array_x.astype(int), array_y.astype(int)
    

def compute_speed_and_acceleration(x:np.array, y:np.array):
    """
    It computes the speed and acceleration of the traits given its x and y coordinates
    
    :param x: the x-coordinates of the points
    :type x: np.array
    :param y: the y-coordinates of the trajectory
    :type y: np.array
    :return: the speed and acceleration of the object.
    """
   
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(np.arange(len(x))) 
    v = np.sqrt(dx**2 + dy**2) / dt
    dv = np.diff(v)
    a = dv / dt[:-1]
    return v,a

    
def create_image_from_array(x:np.array, y:np.array) -> None:
    """
    It takes two arrays as input, merges them, and then draws a line between each point in the array on a Canvas
    
    :param x: The x-coordinates of the points to be plotted
    :type x: np.array
    :param y: The y-coordinates of the points
    :type y: np.array
    """
    
    #Canvas dimensions
    height = 1080 
    width = 1920
    thickness = 2
    color = [0,0,0]
    
    # Create Image Matrix
    image = np.zeros((height, width, 3), np.uint8)
    
    # Fill the Image with white color
    image.fill(255)
    
    # Merge 2 array 
    points = np.column_stack((x, y))
    
    # print(points)
    
    # Loop through all the points for drawing on the canvas
    for i in range(1, len(points)):
        start = tuple(points[i - 1])
        end = tuple(points[i])
        cv2.line(image, start, end, color, thickness)
    
    # Show the image in a new Window
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)