# Import OS for Traversing Directories
import os
# Import Numpy for Mathematical Functions
import numpy as np

# Function to Obtain a List of Files
def getFileList(directory):
    fileList = []
    fileSize = 0
    folderCount = 0
    # For Loop to Cycle Through Directories 
    for root, dirs, files in os.walk(directory):
        folderCount += len(dirs)
        for file in files:
            f = os.path.join(root,file)
            fileSize = fileSize + os.path.getsize(f)
            fileList.append(f)
    # Print to Check Data has been located correctly
    print('################################################')
    print('############ Traversing Directories ############\n')
    print("Total Size is {0} bytes".format(fileSize))
    print('Number of Files = %i') % (len(fileList))
    print('Total Number of Folders = %i') % (folderCount)
    print('################################################\n')
    return fileList

# Location of File
file_loc = '/home/dan/Desktop/IMN432-CW01/meta/'

files = getFileList(file_loc)

numFiles = len(files)

# Import Module
from xml.dom import minidom
from xml.dom.minidom import parseString
import re

table = []
for direc in np.arange(0,numFiles):
    
    # Read and Convert XML to String
    file = open(files[direc],'r')
    data = file.read()
    file.close()
    
    # Ebook Number Retrieval
    ebook = parseString(data)
    ebook = ebook.getElementsByTagName('pgterms:ebook')[0].toxml()
    ebook = ebook.replace('<pgterms:ebook>','').replace('</pgterms:ebook>','')
    ebook = ebook[:ebook.find('>\n')]
    ebook = str(re.split('/',re.split('=',ebook)[1])[1]).replace('"','')
    
    # Read XML into an Object
    xmldoc = minidom.parse(files[direc])    
    
    # Navigate onto the Tree to Find the Ebook No.
    pgterms_ebook = xmldoc.getElementsByTagName('pgterms:ebook')[0]
    
    # Navigate to the Subjects Tree
    subjects = pgterms_ebook.getElementsByTagName('dcterms:subject')

    # Print the Subjects values
    for lines in subjects:
        values = lines.getElementsByTagName('rdf:value')[0].firstChild.data
        table.append([ebook,str(values)])

f = open('Ebook_Subject.txt', 'w')  
for i in np.arange(0,len(table)):
    f.write("%s\n" % table[i])
    
f.close()
