import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale


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


def dataframe_column_to_array(df:pd.DataFrame, column_name:str) -> np.array:
    """
    > This function takes a dataframe and a column name and returns a numpy array of the values in that
    column
    
    :param df: The dataframe that contains the data
    :type df: pd.DataFrame
    :param column_name: The name of the column you want to convert to an array
    :type column_name: str
    :return: A numpy array
    """
    return df[column_name].to_numpy()


def remove_offset_points(x_array:np.array, y_array:np.array):
    """
    It takes in two arrays, subtracts 0 from each element in the array, and returns the two arrays
    
    :param x_array: the x-coordinates of the points
    :type x_array: np.array
    :param y_array: the y-coordinates of the points
    :type y_array: np.array
    :return: The x and y arrays with the offset points removed.
    """
    
    x_array = x_array - 0
    
    y_array = y_array - 0
    
    return x_array, y_array
    

# def scaling_array(array:np.array, max_range:int) -> np.array:
#     """
#     It takes an array, subtracts the minimum value from each element, divides each element by the
#     maximum value minus the minimum value, and then multiplies each element by 255
    
#     :param array: the array to be scaled
#     :type array: np.array
#     :return: The array is being returned.
#     """
    
#     array = ((array - array.min()) * (1/(array.max() - array.min()) * max_range)).astype('uint8')
    
#     return array


def scaling_array(array_x:np.array, array_y:np.array, x_factor:int, y_factor:int ):
    
    array_x = array_x * x_factor/29400
    
    array_y = array_y * y_factor/16600
    
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