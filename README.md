# ShiftFinder

This is the code associated with the analysis presented in the paper: "A method to detect abrupt shifts in river channel position using a Landsat derived water occurrence record"

The data files necessary to run the analysis as presented in the paper can be found at https://zenodo.org/record/6857780#.YtlxbHbMJFQ

## Main analysis pipeline

"CreateOnDiskDataset.ipynb" takes the files in the data folder "EarthEngineAltiplano" and collates them together into one contiguous array so that they can be further analyzed. This notebook also crops the mask to the test area if this is necessary.

"RunTiles.ipynb" creates the A_i map for the test region created in the "CreateOnDiskDataset.ipynb" notebook.

"VizResults.ipynb" uses the A_i map created in "RunTiles.ipynb" to produce most of the rest of the results presented in the paper.

Each of these notebooks cells are commented and there is also some documentation in each notebook to provide a broader overview of what individual cells are doing. "CreateOnDiskDataset.ipynb" and "RunTiles.ipynb" can be run with a small test area as a way to demonstrate the method.

## Other analysis and code

Avul.py is a python module that contains all the custom helper functions associated with the method itself. This includes the main function to compute the A_i value for a given region of interest.

"RunTestSuite.ipynb" loads the data in the folder "TestCases" and computes the A_i value for each of the nine test cases in the test suite. Some of these test cases correspond to what is shown in Figure 7 of the paper.  "Avulsion1YearlyNonGeoCr.tif" is displayed in figure 7F of the paper. "Avulsion3YearlyNonGeoCr.tif" is displayed in figure 7G of the paper. "Avulsion4YearlyNonGeoCr.tif" is displayed in figure 7H of the paper. "BraidedControlYearlyNonGeoCr.tif" is displayed in figure 7E of the paper. The other panels in figure 7 were replaced in the test suite with other control images obtained from actual water surface observations that are labeled "Control" in the test data title. With the exception of the "CutOffControlYearlyNonGeoCr.tif" file they should give an Ai value of one.

"FindMaskedOutAvulsions.ipynb" compares the results of the method and the masking procedure used in the paper to avulsions that were previously found in the study area by hand.

"loadregionandmask.py" is a helper script that loads in the study area/small test area for the "RunTiles.ipynb" and "VizResults.ipynb" notebooks.

## Library versions

Results produced by the code were produced with the packages and versions specified on the "requirements.txt" file. Use python 3.7 - python 3.8 to create a venv to install these specific versions in. Create and install a venv for working with the code by navigating to your local copy of the ShiftFinder directory nad running:
 
```
python3.7 -m venv shiftenv
source shiftenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Computational requirements

To run the analysis on the full test region presented in the paper requires a computer with >32 gb of memory.

"CreateOnDiskDataset.ipynb" and "RunTiles.ipynb" can be run with a small test area as a way to demonstrate the method on computers with 8-16 gb of memory. "RunTestSuite.ipynb" and "FindMaskedOutAvulsions.ipynb" can be run on a computer with 16 gb of memory.
