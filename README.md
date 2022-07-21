# ShiftFinder

This is the code associated with the analysis presented in the paper: "A method to detect abrupt shifts in river channel position using a Landsat derived water occurrence record"

The data files necessary to run the analysis as presented in the paper can be found at https://zenodo.org/record/6857780#.YtlxbHbMJFQ

"CreateOnDiskDataset.ipynb" takes the files in the data folder "EarthEngineAltiplano" and collates them together into one contiguous array so that they can be further analyzed. This notebook also crops the mask to the test area if this is necessary.

Avul.py is a python module that contains all the custom functions assocaited with the method. This includes the main function to compute the A_i value for a given region of interest.

"RunTestSuite.ipynb" loads the data in the folder "TestCases" and computes the A_i value for each of the nine test cases shown in figure 7.

"RunTiles.ipynb" creates the A_i map for the test region

"VizResults.ipynb" uses this map to produce the rest of the results presented in the paper.
