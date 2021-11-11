#!/bin/bash

unzip Inbody_Segmentation_testset_Participant.zip -d ./original
cd original
find . -type f -name "*.jpg" -exec mv {} ../val2014 \;
find . -type f -name "*.png" -exec mv {} ../val_mask \;
rm -r original