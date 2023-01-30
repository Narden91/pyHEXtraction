import hydra
import pandas as pd
import re
import os
import csv
import preprocessing 


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    
    # Get all the folders (complete PATH) inside data (to get only folder name replace f.path with f.name)
    folders_in_data = [f.path for f in os.scandir(config.data_source) if f.is_dir()]

    # Debug variable
    count_var = True
    
    for folder in folders_in_data:
        
        # Debug compute only the first folder of the folder list
        if count_var:    
            # Get all the .csv files (COMPLETE PATH) inside the folder 
            file_list = [f.path for f in os.scandir(folder) if f.is_file() and f.path.endswith(config.file_extension)]
            
            # Sort the file based on filename
            file_list = sorted(file_list, key = lambda x: len(x))
            
            for file in file_list:
                
                # Debug compute only the first file in the folder
                if count_var:     
                    
                    print(f"Filename: {file} \n")
                    
                    file_dataframe = preprocessing.dataframe_from_csv(file)
                    
                    print(f"Dataframe: \n {file_dataframe} \n")
                    
                    #file_dataframe = preprocessing.filter_dataframe_rows_by_value(file_dataframe, 'Phase', 'MoveStroke')
                                        
                    #print(f"Dataframe: \n {file_dataframe} \n")
  
                    count_var = False
                else:
                    break  
        else:
            break        
    
    return

if __name__== '__main__':
    main()
    
    