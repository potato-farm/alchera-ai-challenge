#!/bin/bash

mkdir -p original
unzip Inbody_Segmentation_testset_Participant_pixelchange_gray-smooth.zip -d ./original
mkdir -p val_mask
mkdir -p val2014
cd original
find . -type f -name "*.jpg" -exec mv {} ../val2014 \;
find . -type f -name "*.png" -exec mv {} ../val_mask \;
cd ..
rm -r original
rm Inbody_Segmentation_testset_Participant_pixelchange_gray-smooth.zip