#!/bin/bash

# from https://github.com/merlresearch/SMART

wget https://zenodo.org/record/7775984/files/SMART101-release-v1.zip -P data
cd data
unzip SMART101-release-v1.zip -d smart-data

# fixing an error in the dataset
mv smart-data/SMART101-release-v1/SMART101-Data/62/puzzle_63.csv smart-data/SMART101-release-v1/SMART101-Data/62/puzzle_62.csv