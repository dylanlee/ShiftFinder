import os
import numpy as np
from skimage import io
import avul  # Assuming avul is a module containing YearBinarize function

# Define a global variable to store the processed images
ImList = []

# Define a function to import test datasets and process them
def import_test_datasets():
    global ImList
    # Define the file paths for the test datasets
    file_paths = [
        "./data/TestCases/Avulsion1YearlyNonGeoCr.tif",
        "./data/TestCases/Avulsion2YearlyNonGeoCr.tif",
        "./data/TestCases/Avulsion3YearlyNonGeoCr.tif",
        "./data/TestCases/Avulsion4YearlyNonGeoCr.tif",
        "./data/TestCases/Avulsion5YearlyNonGeoCr.tif",
        "./data/TestCases/MeanderControlYearlyNonGeoCr.tif",
        "./data/TestCases/BranchingControlYearlyNonGeoCr.tif",
        "./data/TestCases/BraidedControlYearlyNonGeoCr.tif",
        "./data/TestCases/CutOffControlYearlyNonGeoCr.tif"
    ]

    # Load images and apply YearBinarize
    ImList = []
    for file_path in file_paths:
        im = io.imread(file_path)
        BinData = avul.YearBinarize(im)
        ImList.append(BinData)

# Import the datasets when the module is loaded
import_test_datasets()

# Make ImList accessible when importing the script
__all__ = ['ImList', 'import_test_datasets']
