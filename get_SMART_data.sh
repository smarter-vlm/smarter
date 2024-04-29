#!/bin/bash

# from https://github.com/merlresearch/SMART

wget https://zenodo.org/record/7775984/files/SMART101-release-v1.zip -P data
cd data
unzip SMART101-release-v1.zip -d smart_data