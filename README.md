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

"RunTestSuite.ipynb" loads the data in the folder "TestCases" and computes the A_i value for each of the nine test cases shown in figure 7.

"FindMaskedOutAvulsions.ipynb" compares the results of the method and the masking procedure used in the paper to avulsions that were previously found in the study area by hand.

"loadregionandmask.py" is a helper script that loads in the study area/small test area for the "RunTiles.ipynb" and "VizResults.ipynb" notebooks.

## Library versions

Code was created using the following versions of python and main libraries:

- Python 3.8.8
- Numpy 1.20.1
- Rasterio 1.3.0
- Dask 2021.04.0
- Scipy 1.6.2
- Matplotlib 3.3.3
- h5py 2.10.0
- numba 0.53.1
- skimage 0.18.1
- sklearn 0.24.1

## Computational requirements

To run the analysis on the full test region presented in the paper requires a computer with >32 gb of memory.

"CreateOnDiskDataset.ipynb" and "RunTiles.ipynb" can be run with a small test area as a way to demonstrate the method on computers with 8-16 gb of memory. "RunTestSuite.ipynb" and "FindMaskedOutAvulsions.ipynb" can be run on a computer with 16 gb of memory.
