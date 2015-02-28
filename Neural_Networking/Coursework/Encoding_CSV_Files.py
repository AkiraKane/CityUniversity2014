## Preparation of the Data
import pandas as pd

# Import the Data
bank_data = pd.read_csv('/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/IMN427_Neural_Computing/Week 5/lab/bank-additional-full.csv', sep=';')

# Check the Data was Imported correctly
bank_data.head(3)

# Get Numberic Columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Bank_Data_Numerics = bank_data.select_dtypes(include=numerics)
numeric_col = Bank_Data_Numerics.columns

# Get Categorial Columns
Cat_col_names = bank_data.columns - numeric_col
bank_data_Cat = bank_data[Cat_col_names]

Mapping_Table = pd.DataFrame()

# Loop Through the Categorical Table and Convert to Numerics
for col in Cat_col_names:
    # Convert String to Integer
    bank_data[col] = pd.Categorical.from_array(bank_data[col]).codes
    # Join Togeather and create a Mapping Table
    Mapping_Table = Mapping_Table.append(pd.concat([bank_data_Cat[col], bank_data[col]],axis=1,ignore_index=True).drop_duplicates())

# Correct the Dataframe Headings
Mapping_Table.columns = ['Label','Numerical_Code']

# Save Encoded Dataframes
bank_data.to_csv('/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/IMN427_Neural_Computing/Week 5/lab/bank-additional-full-encoded.csv', sep=',', index=False)
Mapping_Table.to_csv('/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/IMN427_Neural_Computing/Week 5/lab/bank-additional-full-mapping-table.csv', sep=',', index=False)