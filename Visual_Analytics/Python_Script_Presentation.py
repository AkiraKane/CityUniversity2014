# Import Modules
import pandas as pd
import numpy as np

# Run the main functions
def main():
    # Import Data
    Data = import_data()
    # Selecting Specific Columns
    Data_Components = processingData(Data)
    
# Import Function
def import_data():
    # Print Statement
    print('Importing Data')
    # File path
    filePath = '/home/dan/Desktop/Github_Files/Visual_Analytics/'
    # Import the Data
    Data = pd.read_csv(filePath + 'Crossfit_Open_2011_Dataset.csv', sep = ',')
    # Print Shape    
    print('Importing Complete\n')
    # Return Data to Main
    return Data

def processingData(DF):
    # Print Statement
    print('Processing Data')
    Col_Names = DF.columns
    # Columns to Exclude
    Col_Excl =[u'athlete_ID', u'nameURL', u'First_Name', u'Last_Name', u'Region',u'sex&division',u'Gender',u'height',u'Weight_Orginal',u' overall-points', u'overall-rank',u'rank1',u' rank2',u'rank3',u'rank4',u'rank5',u'rank6']
    # Subset by Gender - Male
    Male =  DF[DF.Gender == 'M']
    # Subset by Gender - Female
    Female =  DF[DF.Gender == 'F']
    
    # Return Data to Main
    return DF_Components
    

if __name__ == "__main__":
    # Run the Algorithm    
    main()