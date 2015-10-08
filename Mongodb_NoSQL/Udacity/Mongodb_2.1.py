# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:09:17 2015

@author: dan
"""

from pymongo import MongoClient
import pprint

client = MongoClient("mongodb://localhost:27017")

db = client.examples


def find():
    autos = db.autos.find(
        {"manufacturer": "Toyota",
         "class": "mid-size car"})
    for a in autos:
        pprint.pprint(a)

if __name__ == '__main__':
    find()
