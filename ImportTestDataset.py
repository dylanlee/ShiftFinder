#Import monthly data stack
#Avulsion 1
im1 = io.imread("./GlobalAvulsionData/Avulsion1YearlyNonGeoCr.tif")
#Avulsion 2
im2 = io.imread("./GlobalAvulsionData/Avulsion2YearlyNonGeoCr.tif")
#Avulsion 3
im3 = io.imread("./GlobalAvulsionData/Avulsion3YearlyNonGeoCr.tif")
#Avulsion 4
im4 = io.imread("./GlobalAvulsionData/Avulsion4YearlyNonGeoCr.tif")
#Avulsion 5
im5 = io.imread("./GlobalAvulsionData/Avulsion5YearlyNonGeoCr.tif")
#Meandering Control
im6 = io.imread("./GlobalAvulsionData/MeanderControlYearlyNonGeoCr.tif")
#Branching Control
im7 = io.imread("./GlobalAvulsionData/BranchingControlYearlyNonGeoCr.tif")
#Braided Control
im8 = io.imread("./GlobalAvulsionData/BraidedControlYearlyNonGeoCr.tif")
#Cutoff Control
im9 = io.imread("./GlobalAvulsionData/CutOffControlYearlyNonGeoCr.tif")

#add im -im5 to a dictionary
ImList = [im1,im2,im3,im4,im5,im6,im7,im8,im9]
for x in range(9):
        BinData = avul.YearBinarize(ImList[x])
        #apply median filter to image. Getting rid of this step for now
        #BinData = median_filter(BinData,2)
        ImList[x] = BinData
        
