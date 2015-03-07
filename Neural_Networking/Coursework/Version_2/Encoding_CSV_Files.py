## Preparation of the Data
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV

# Import the Data
bank_data = pd.read_csv('/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/CityUniversity2014/Neural_Networking/Coursework/Version_2/bank-additional-full.csv', sep=';')

# Check the Data was Imported correctly
bank_data.head(3)

# Get Numberic Columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Bank_Data_Numerics = bank_data.select_dtypes(include=numerics)
numeric_col = Bank_Data_Numerics.columns

# Get Categorial Columns
Cat_col_names = bank_data.columns - numeric_col
bank_data_Cat = bank_data[Cat_col_names]

# Get a dictionary for the transformation
dict_DF = bank_data_Cat.T.to_dict().values()

# Vectorizer
vectorizer = DV( sparse = False )

# Transform Dataset
Dataset_Binary = vectorizer.fit_transform( dict_DF )

# Get the Revised Column Names
New_Colnames = vectorizer.get_feature_names()

# Convert Dataset Binary to a Dataframe
Dataset_Binary_DF = pd.DataFrame(Dataset_Binary)

# Add columns Names
Dataset_Binary_DF.columns = New_Colnames

# Drop the Last two Columns
Dataset_Binary_DF = Dataset_Binary_DF.ix[:,0:52]

# Convert the Binary Yes No to binary values
Transformed_Target = pd.Categorical.from_array(bank_data['y']).codes

# Convert the code to a dataframe
Transformed_Target_DF = pd.DataFrame(Transformed_Target)

# Add the column Names - as it was lost in the transformation
Transformed_Target_DF.columns = ['y']

# Concat all the Dataframes
Finished_DF = pd.concat([Dataset_Binary_DF, Transformed_Target_DF], axis=1)

# Save Encoded Dataframes
Finished_DF.to_csv('/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/CityUniversity2014/Neural_Networking/Coursework/Version_4/bank-additional-full-binary-transform.csv', sep=',', index=False)