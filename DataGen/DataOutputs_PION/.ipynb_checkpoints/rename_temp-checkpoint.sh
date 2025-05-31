#!/bin/bash

# Set the directory path

# Loop through all files ending with 'csv' in the directory
for file in EnSplit/*csv; do
    # Get the filename without the extension
    filename=$(basename "$file" .csv)
    filename=${filename:0:14}
    # Rename the file by inserting a dot before the 'csv' extension
    mv "$file" "EnSplit/$filename.csv"
done