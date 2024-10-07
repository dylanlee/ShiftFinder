#!/usr/bin/env python
import numpy as np
from scipy import ndimage
from TestDataset import ImList
import pyvista as pv

import avul

# Define getpoints function
def getpoints(im):
    WaterLoc = np.nonzero(im)
    Points = np.array([WaterLoc[0], WaterLoc[1], WaterLoc[2]]).T
    return Points

# Ask the user to input a test case number
test_case_number = int(input(f"Please enter a test case number (1 to {len(ImList)}): "))

# Adjust for zero-based indexing
index = test_case_number - 1

# Check if the index is within the valid range
if index < 0 or index >= len(ImList):
    print(f"Invalid test case number. Please enter a number between 1 and {len(ImList)}.")
else:
    # Get the image corresponding to the test case number
    Im = ImList[index]

    # Get the connected components in the image
    labels_out, _ = ndimage.label(Im, structure=np.ones((3, 3, 3)))
    FlLab = labels_out.flatten()
    FlLab = FlLab[FlLab > 0]

    if FlLab.size > 0:
        # Find the largest connected component
        u, indices = np.unique(FlLab, return_inverse=True)
        BigComLab = u[np.argmax(np.bincount(indices))]
        ComIm = np.where(labels_out == BigComLab, labels_out, 0)
        num_points = np.count_nonzero(ComIm)
        print(f"Number of points in the largest connected component: {num_points}")

        # Get the points from the connected component
        points = np.argwhere(ComIm > 0)

        # Create a PyVista PolyData object
        point_cloud = pv.PolyData(points)

        # Visualize using PyVista
        plotter = pv.Plotter()
        plotter.add_mesh(point_cloud, color='blue', point_size=2.0, render_points_as_spheres=True)
        plotter.show_axes()
        plotter.show(title=f'Connected Component Point Cloud (Test Case {test_case_number})')
    else:
        print("No connected components found in the image.")
