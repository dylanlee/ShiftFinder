import h5py
import avul
import dask as da
from dask import array

#use line below once to load in the test dataset
#need to use the 'r' option instead of 'a' for dask distributed to work
hf = h5py.File('../Data/TestRegion.h5','r')
d = hf['./Altiplano']          # Pointer on on-disk array

#convert the on-disk array to a dask array
Im = array.from_array(d,chunks='auto')

#this loads your selection into memory
Im = Im.astype('uint8') #convert to unsigned int 8 to save space
Im = Im.compute()

#load in mask
mask = plt.imread('../Data/TestRegionMask.png')
mask[mask == 255] = 1
mask = mask[:,:,1] #only need one channel since binary

#Binarize the image. Only run this once after loading image into memory!
Im = avul.YearBinarize(Im)

#get windows for rolling window analysis
strideparams = [60,365] #1st value is step size, 2nd value is window size
WinShape = Im.shape[1:3]
SubWins = avul.GetSubWins(WinShape,strideparams)
ZlevImShape = SubWins.shape[0:2]
ZWin = np.ones(ZlevImShape).astype(int)
ZlevFlatIn = np.cumsum(ZWin)
ZlevFlatIn = np.subtract(ZlevFlatIn,1)
#now iterate over sub-windows to build up image after taking the median at each window
SearIm = np.zeros(ZlevImShape)
ActLevIm = np.zeros(ZlevImShape)

#make sure your subwin shapes are an even divisor of your main image size
step = strideparams[0]
size = strideparams[1]
print((Im.shape[1] - size + 1) / step)
print((Im.shape[2] - size + 1) / step)