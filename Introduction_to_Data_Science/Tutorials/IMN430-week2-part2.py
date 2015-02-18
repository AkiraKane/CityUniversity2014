####
# Tutorial Week2 - Pandas
import pandas as pd
#    Download the two files and load them into pandas data frames. Hint: have a look at this page and do not forget to parse your excel sheet into a data frame.
#    Merge the two files based on the column they share.
#    Display the name of the oldest passengers (hint: make use of variables to save some intermediate values).
#    Plot the data on a scatter plot that shows the Age vs. Ticket Prices
#    Plot only the data that shows female passengers aged 40 to 50 and who paid more than or equal to 40.

##### 1
## Posible to Download directly from URL when using the read_csv function
Passenger_Data = pd.read_csv("http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/passengerData.csv")
print Passenger_Data.columns
## Not Possible with read xlsx, using StringIO and urllib2
import urllib2
import StringIO
url = "http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/ticketPrices.xlsx"
xld = urllib2.urlopen(url).read()
xld = StringIO.StringIO(xld)
# Finally Read Data into Pandas Dataframe
Ticket_Prices = pd.read_excel(xld, "Sheet1")
print Ticket_Prices.columns

##### 2
# Merge Dataframes on TicketType
# Left Join Data
Complete_Data_Set = pd.merge(Passenger_Data, Ticket_Prices, how='left', left_on="TicketType", right_on="TicketType")
print Complete_Data_Set

##### 3
print "Max age: %i" % (Complete_Data_Set['Age'].max())
print "Passenger Details:"
print Complete_Data_Set[Complete_Data_Set['Age'] == Complete_Data_Set['Age'].max()]

##### 4
import matplotlib.pyplot as plt
plt.scatter(Complete_Data_Set['Age'], Complete_Data_Set['Fare'])
plt.title('Age vs. Ticket Prices')
plt.xlabel("Passenger's Age")
plt.ylabel("Ticket Price")
plt.show()

##### 5
# Example I learnt from to filter on multiple columns df[(df['A'] > 1) | (df['B'] < -1)], dont forget the brackets...
Subset = Complete_Data_Set[(Complete_Data_Set['Sex']=="female") & (Complete_Data_Set['Age'] >= 40) & (Complete_Data_Set['Age'] <= 50) & (Complete_Data_Set['Fare'] >= 40)]
plt.scatter(Subset['Age'], Subset['Fare'])
plt.title('Age vs. Ticket Prices')
plt.xlabel("")
plt.ylabel("Ticket Price")
plt.show()
