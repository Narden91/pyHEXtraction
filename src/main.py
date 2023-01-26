import hydra
import pandas as pd
import re
import os
import preprocessing 


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    
    # Get all the folders (complete PATH) inside data (to get only folder name replace f.path with f.name)
    folders_in_data = [f.path for f in os.scandir(config.data_source) if f.is_dir()]

    count_var = True
    
    for folder in folders_in_data:
        
        # Debug
        if count_var:    
            # Get all the .csv files (COMPLETE PATH) inside the folder 
            # file_list = [f.path for f in os.listdir(folder) if '.csv' in f]
            file_list = [d.path for d in os.scandir(folder) if d.is_file() and d.path.endswith('csv')]
            
            # Sort the file based on filename
            file_list = sorted(file_list, key = lambda x: len(x))
            
            for file in file_list:
                
                # Debug
                if count_var:     
                    
                    print(f"Filename: {file}")
                              
                    csv_dataframe = pd.read_csv(file, delimiter=":")
                    
                    print(f"Dataframe: {csv_dataframe}")
                    
                    count_var = False
                else:
                    break  
        else:
            break        
    
    return

if __name__== '__main__':
    main()
    
    