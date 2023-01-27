import pandas as pd



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
                
    print(df)
  
    return df
