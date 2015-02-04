# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 15:14:30 2014

@author: ackf415
"""
# XML Parsing
from bs4 import BeautifulSoup
# Web Page Scraper
import requests
# Creating XML Documents
import xml.etree.ElementTree as ET
# get the URL into a variable
urlToScrape = "http://www.amazon.co.uk/product-reviews/1407109081/ref=cm_cr_dp_see_all_btm?ie=UTF8&showViewpoints=1&sortBy=bySubmissionDateDescending"
# request the web page
r  = requests.get(urlToScrape)
# convert it into text
data = r.text
# create a Beautiful Soap parser
soup = BeautifulSoup(data, "lxml")

###################################################
### DIY Exercises - 1 : Scrape Data From the Web ##
###################################################
print(soup.prettify())

# 1 - List and print all the addresses of all the 
#     links on this page. Hint: use soup.find_all() 
#     and soup.get() functions.

# Find all the Link URL tags
allLinks = soup.find_all('a')
# For loop to iterate through each item in List
for link in allLinks:
    print(link.get('href'))
    
# 2 - List all the dates of the reviews as text.

# Get Table Data where the Reviews reside
Dates = []
for name in soup.find_all('nobr'):
    if '2' in name.text:
        Dates.append([name.text])

# 3 - List all the reviews.
Reviews = []
for reviews in soup.find_all("div", "reviewText"):
    Reviews.append([reviews.text])

# 4 - Decide on a proper XML structure and save them
#     on an XML file, you can use the XML generation
#      functions in Python.

# Import the XML Building Tool

XML_Doc = ET.Element('Body')
for i in Dates:
    b = ET.SubElement(XML_Doc, str(i))
    for j in Reviews:
        c = ET.SubElement(b, str(j))
Output = ET.dump(XML_Doc)

# 5 - You may have noticed that the page we've 
#     selected do not display all the comments, 
#     only a portion of it. One needs to iterate 
#     through the review pages one by one. Inspect 
#     the Amazon URL and find a way to iterate over 
#     the comment pages.


# 6 - Parse the first 10, 20, or all the review 
#     pages and collect all the comments with their 
#     dates on an XML file.
