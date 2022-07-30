
#Collection of repeatedly called code to get it out of the notebook workspace
import numpy as np
from numpy import inf
import random
from scipy import ndimage
import math
import skimage.measure
import sklearn.neighbors
import warnings
from numba import jit as numba_jit
import numba
from IPython.core.debugger import set_trace

def getvals(Im,SubMask,StabCrit,IsBinIm):

    #get the biggest connected components in the image
    #if working with a non-binarized image you need to do this.
    if IsBinIm == 0:
        Im = YearBinarize(Im)
    labels_out,tra = ndimage.measurements.label(Im,structure=np.ones((3,3,3)))
    FlLab = np.ravel(labels_out)
    FlLab = FlLab[np.nonzero(FlLab)]
    CurseaR = 0
    NumActLevels = 0
    ActLevels = 0
    finComIm = np.zeros((labels_out.shape))
    
    if FlLab.size>0:
        
        u, indices = np.unique(FlLab, return_inverse=True)
        #select labels that are bigger than %1 percent of the observation volume (assuming 300x300x34yr volume)
        BigComLabs = u[np.nonzero(np.bincount(indices)>30000)]
        #now go through and mask out components that don't have a pixel overlapping with mask
        for x in range(BigComLabs.size):
            
            ComIm = labels_out * (labels_out == BigComLabs[x])
            #flatten along time
            allRed = skimage.measure.block_reduce(ComIm,(35,1,1),np.max)
            maskIm = SubMask * allRed
            #if a pixel overlaps then keep since it is a fluvial component
            if (np.max(maskIm) > 0):
                finComIm = np.logical_or(ComIm,finComIm) 
        #now write the fluvial components back to ComIm for further analysis
        ComIm = finComIm
        #if there are no fluvial components then move on
        if not finComIm.any():
            NumActLevels = 0
            CurseaR = 0
            ActLevels = 0
        else:
            
            #FiltIm = simpfilt(ComIm)
            points = getpoints(ComIm)
            NumActLevels, CurseaR, ActLevels = stbsteps(points,StabCrit)
      
    #print("search radius is " + str(CurseaR))
    #print("Active Levels: " + str(NumActLevels))
    return np.asarray([NumActLevels, CurseaR,ActLevels]).reshape(3,1)
    
def stbsteps(points,StabCrit):

    bt2 = BallTree(points, leaf_size=40)
    LevCts = np.empty(0)
    numActLevels = 35
    AppCount = np.zeros(35)
    Ct = 0
    while numActLevels > 1:
        #print(Ct)
        seaR = np.sqrt(np.square(Ct) + np.square(Ct) + np.square(Ct))
        if Ct == 0:
            seaR = 1 #seaR = 1 is the base case of just going straight up
            EnPtsLoc = bt2.query(points, seaR)
            EnPts = points[EnPtsLoc]
        else:
            EnPtsLoc = bt2.query(EnPts, seaR)
            EnPts = EnPts[EnPtsLoc]
        
        #get number of levels
        Zlevs = EnPts[:,2]
        ActLevels = np.unique(Zlevs)
        numActLevels = ActLevels.size
        #print(seaR)
        #print(numActLevels)
        LevCts = np.append(LevCts,[numActLevels])

        if Ct >= StabCrit:
            #stop increasing search distance if your number of levels has been stable for 'StabCrit' iterations
            if np.unique(LevCts[-StabCrit:]).size == 1 :
                #set_trace()
                #set the seaR you write to the sear at the base of your stabcrit (essentially correcting sear for repeat read criterion)
                seaR = np.sqrt(np.square(Ct-StabCrit+1) + np.square(Ct-StabCrit+1) + np.square(Ct-StabCrit+1))
                break
                
        Ct = Ct + 1
    #if there was only one level on the very 1st try then consider it as not having moved
    if Ct == 1: #Ct == 1 because you add to Ct immediately above
        seaR = 0
    return numActLevels, seaR, ActLevels
    


def getpoints(im):
    WaterLoc = np.nonzero(im)
    Points = np.array([WaterLoc[1],WaterLoc[2],WaterLoc[0]]).T
    return Points

def YearBinarize(im):
    #Binarize data
    im[im == 1] = 0
    im[im == 2] = 1
    im[im == 3] = 1
    return im

def GetSubWins(WinShape,strideparams):
    Win = np.ones(WinShape).astype(int)
    FlatIn = np.cumsum(Win)
    FlatIn = np.subtract(FlatIn,1)
    Win = np.reshape(FlatIn,WinShape)

    ##Stride params
    xstep = int(strideparams[0])
    ystep = int(strideparams[0])
    xsize = int(strideparams[1])
    ysize = int(strideparams[1])

    SubWins = np.lib.stride_tricks.as_strided(Win,(int((Win.shape[0] - xsize + 1) / xstep), int((Win.shape[1] - ysize + 1) / ystep), xsize, ysize),(Win.strides[0] * xstep, Win.strides[1] * ystep, Win.strides[0], Win.strides[1]))
    
    return SubWins

#takes a subim and extracts the primary region from it
def GetReg(SubIm,SubMask):
    #SubIm = avulHPC.YearBinarize(SubIm)
    #get the biggest connected components in the image
    labels_out,tra = ndimage.measurements.label(SubIm,structure=np.ones((3,3,3)))
    #clear SubIm and recast labels_out to save space when doing this for bigger images
    SubIm = []
    labels_out = labels_out.astype('uint16')
    FlLab = np.ravel(labels_out)
    FlLab = FlLab[np.nonzero(FlLab)]
    finComIm = np.zeros((labels_out.shape))
    FlLab = np.ravel(labels_out)
    FlLab = FlLab[np.nonzero(FlLab)]
    u, indices = np.unique(FlLab, return_inverse=True)
    #select labels that are bigger than 10000 elements
    #set_trace()
    BigComLabs = u[np.nonzero(np.bincount(indices)>30000)]
    indices = [] #clear indices to save space again
    #print("number of labels")
    #print(BigComLabs.size)
    #now go through and mask out components that don't have a pixel overlapping with mask
    for x in range(BigComLabs.size):
        ComIm = labels_out * (labels_out == BigComLabs[x])
        ComIm = ComIm.astype('uint8')
        #flatten along time
        allRed = skimage.measure.block_reduce(ComIm,(35,1,1),np.max)
        maskIm = SubMask * allRed
        #if a pixel overlaps and this is a fluvial component then keep.
        if (np.max(maskIm) > 0):
            finComIm = np.logical_or(ComIm,finComIm) 
    #if there is nothing in finComIm then keep as an all zero array
    if np.any(finComIm):
        #change the nonzero pixels in ComIm to 1 to get active areas
        finComIm[finComIm == np.max(finComIm)] = 1
    return finComIm

#### balltree code + endpoint tree query ####

#----------------------------------------------------------------------
# Distance computations

@numba.jit(nopython=True)
def rdist(X1, i1, X2, i2):
    d = 0
    for k in range(X1.shape[1]):
        tmp = (X1[i1, k] - X2[i2, k])
        d += tmp * tmp
    return d

@numba.jit(nopython=True)
def min_rdist(node_centroids, node_radius, i_node, X, j):
    d = rdist(node_centroids, i_node, X, j)
    return np.square(max(0, np.sqrt(d) - node_radius[i_node]))

#----------------------------------------------------------------------
# Heap for distances and neighbors

@numba.jit(nopython=True)
def heap_create(N, k):
    distances = np.full((N, k), np.finfo(np.float64).max)
    indices = np.zeros((N, k), dtype=np.int64)
    return distances, indices

#----------------------------------------------------------------------
# Tools for building the tree

@numba.jit(nopython=True)
def _partition_indices(data, idx_array, idx_start, idx_end, split_index):
    # Find the split dimension
    n_features = data.shape[1]

    split_dim = 0
    max_spread = 0

    for j in range(n_features):
        max_val = -np.inf
        min_val = np.inf
        for i in range(idx_start, idx_end):
            val = data[idx_array[i], j]
            max_val = max(max_val, val)
            min_val = min(min_val, val)
        if max_val - min_val > max_spread:
            max_spread = max_val - min_val
            split_dim = j

    # Partition using the split dimension
    left = idx_start
    right = idx_end - 1

    while True:
        midindex = left
        for i in range(left, right):
            d1 = data[idx_array[i], split_dim]
            d2 = data[idx_array[right], split_dim]
            if d1 < d2:
                tmp = idx_array[i]
                idx_array[i] = idx_array[midindex]
                idx_array[midindex] = tmp
                midindex += 1
        tmp = idx_array[midindex]
        idx_array[midindex] = idx_array[right]
        idx_array[right] = tmp
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1

@numba.jit(nopython=True)
def _recursive_build(i_node, idx_start, idx_end,
                     data, node_centroids, node_radius, idx_array,
                     node_idx_start, node_idx_end, node_is_leaf,
                     n_nodes, leaf_size):
    # determine Node centroid
    for j in range(data.shape[1]):
        node_centroids[i_node, j] = 0
        for i in range(idx_start, idx_end):
            node_centroids[i_node, j] += data[idx_array[i], j]
        node_centroids[i_node, j] /= (idx_end - idx_start)

    # determine Node radius
    sq_radius = 0.0
    for i in range(idx_start, idx_end):
        sq_dist = rdist(node_centroids, i_node, data, idx_array[i])
        if sq_dist > sq_radius:
            sq_radius = sq_dist

    # set node properties
    node_radius[i_node] = np.sqrt(sq_radius)
    node_idx_start[i_node] = idx_start
    node_idx_end[i_node] = idx_end

    i_child = 2 * i_node + 1

    # recursively create subnodes
    if i_child >= n_nodes:
        node_is_leaf[i_node] = True
        if idx_end - idx_start > 2 * leaf_size:
            # this shouldn't happen if our memory allocation is correct.
            # We'll proactively prevent memory errors, but raise a
            # warning saying we're doing so.
            #warnings.warn("Internal: memory layout is flawed: "
            #              "not enough nodes allocated")
            pass

    elif idx_end - idx_start < 2:
        # again, this shouldn't happen if our memory allocation is correct.
        #warnings.warn("Internal: memory layout is flawed: "
        #              "too many nodes allocated")
        node_is_leaf[i_node] = True

    else:
        # split node and recursively construct child nodes.
        node_is_leaf[i_node] = False
        n_mid = int((idx_end + idx_start) // 2)
        _partition_indices(data, idx_array, idx_start, idx_end, n_mid)
        _recursive_build(i_child, idx_start, n_mid,
                         data, node_centroids, node_radius, idx_array,
                         node_idx_start, node_idx_end, node_is_leaf,
                         n_nodes, leaf_size)
        _recursive_build(i_child + 1, n_mid, idx_end,
                         data, node_centroids, node_radius, idx_array,
                         node_idx_start, node_idx_end, node_is_leaf,
                         n_nodes, leaf_size)
                         
#----------------------------------------------------------------------
# Tools for querying the tree
@numba.jit(nopython=True)
def _query_recursive(i_node, X, seaR, i_pt, heap_distances, heap_indices, sq_dist_LB,
                     data, idx_array, node_centroids, node_radius,
                     node_is_leaf, node_idx_start, node_idx_end):
    #------------------------------------------------------------
    # Case 1: query point is outside node radius:
    #         trim it from the query
    if np.sqrt(sq_dist_LB) > seaR:
        pass

    #------------------------------------------------------------
    # Case 2: this is a leaf node.  Update set of nearby points
    elif node_is_leaf[i_node]:
        for i in range(node_idx_start[i_node],
                       node_idx_end[i_node]):
            dist_pt = rdist(data, idx_array[i], X, i_pt)
            #if point is less than seaR away write this as an endpoint
            if np.sqrt(dist_pt) < seaR:
                #if point has a neigbor above it change index to 2 to mark (normal point)
                if data[idx_array[i]][2] > X[i_pt][2]:
                    heap_indices[i_pt][0] = 2
                if heap_indices[i_pt,0] != 2:#heap_distances[i_pt, 0]:                    
                    heap_indices[i_pt,0] = 1

    #------------------------------------------------------------
    # Case 3: Node is not a leaf.  Recursively query subnodes
    #         starting with the closest
    else:
        i1 = 2 * i_node + 1
        i2 = i1 + 1
        sq_dist_LB_1 = min_rdist(node_centroids,
                                 node_radius,
                                 i1, X, i_pt)
        sq_dist_LB_2 = min_rdist(node_centroids,
                                 node_radius,
                                 i2, X, i_pt)

        # recursively query subnodes
        if sq_dist_LB_1 <= sq_dist_LB_2:
            _query_recursive(i1, X, seaR, i_pt, heap_distances,
                             heap_indices, sq_dist_LB_1,
                             data, idx_array, node_centroids, node_radius,
                             node_is_leaf, node_idx_start, node_idx_end)
            _query_recursive(i2, X, seaR, i_pt, heap_distances,
                             heap_indices, sq_dist_LB_2,
                             data, idx_array, node_centroids, node_radius,
                             node_is_leaf, node_idx_start, node_idx_end)
        else:
            _query_recursive(i2, X, seaR, i_pt, heap_distances,
                             heap_indices, sq_dist_LB_2,
                             data, idx_array, node_centroids, node_radius,
                             node_is_leaf, node_idx_start, node_idx_end)
            _query_recursive(i1, X, seaR, i_pt, heap_distances,
                             heap_indices, sq_dist_LB_1,
                             data, idx_array, node_centroids, node_radius,
                             node_is_leaf, node_idx_start, node_idx_end)

@numba.jit(nopython=True, parallel=True)
def _query_parallel(i_node, X, seaR, heap_distances, heap_indices,
                     data, idx_array, node_centroids, node_radius,
                     node_is_leaf, node_idx_start, node_idx_end):
    for i_pt in numba.prange(X.shape[0]-1):
        sq_dist_LB = min_rdist(node_centroids, node_radius, i_node, X, i_pt)
        _query_recursive(i_node, X, seaR, i_pt, heap_distances, heap_indices, sq_dist_LB,
                         data, idx_array, node_centroids, node_radius, node_is_leaf,
                         node_idx_start, node_idx_end)

#----------------------------------------------------------------------
# The Ball Tree object
class BallTree(object):
    def __init__(self, data, leaf_size=40):
        self.data = data
        self.leaf_size = leaf_size

        # validate data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")

        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]

        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points betweeen leaf_size and 2 * leaf_size
        self.n_levels = 1 + np.log2(max(1, ((self.n_samples - 1)
                                            // self.leaf_size)))
        self.n_nodes = int(2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(self.n_samples, dtype=int)
        self.node_radius = np.zeros(self.n_nodes, dtype=float)
        self.node_idx_start = np.zeros(self.n_nodes, dtype=int)
        self.node_idx_end = np.zeros(self.n_nodes, dtype=int)
        self.node_is_leaf = np.zeros(self.n_nodes, dtype=int)
        self.node_centroids = np.zeros((self.n_nodes, self.n_features),
                                       dtype=float)

        # Allocate tree-specific data from TreeBase
        _recursive_build(0, 0, self.n_samples,
                         self.data, self.node_centroids,
                         self.node_radius, self.idx_array,
                         self.node_idx_start, self.node_idx_end,
                         self.node_is_leaf, self.n_nodes-1, self.leaf_size)

    def query(self, X, seaR, sort_results=True):
        X = np.asarray(X, dtype=float)

        if X.shape[-1] != self.n_features:
            raise ValueError("query data dimension must "
                             "match training data dimension")

        #if self.data.shape[0] < k:
        #    raise ValueError("k must be less than or equal "
        #                     "to the number of training points")

        # flatten X, and save original shape information. This seems to be for
        # when you have stacks of observations of a given dimension
        Xshape = X.shape
        X = X.reshape((-1, self.data.shape[1]))

        # initialize heap for neighbors
        heap_distances, heap_indices = heap_create(X.shape[0], 1)

        #for i in range(X.shape[0]):
        #    sq_dist_LB = min_rdist(self.node_centroids,
        #                           self.node_radius,
        #                           0, X, i)
            #_query_recursive(0, X, i, heap_distances, heap_indices, sq_dist_LB,
            #                 self.data, self.idx_array, self.node_centroids,
            #                 self.node_radius, self.node_is_leaf,
            #                 self.node_idx_start, self.node_idx_end)

        _query_parallel(0, X, seaR, heap_distances, heap_indices,
                     self.data, self.idx_array, self.node_centroids, self.node_radius,
                     self.node_is_leaf, self.node_idx_start, self.node_idx_end)
        
        #points that don't transition are endpoints
        heap_indices[heap_indices == 0] = 1
        #all normal points (2) got sent to 0
        heap_indices[heap_indices == 2] = 0
        EnPtsLoc = np.nonzero(heap_indices)[0]
        return EnPtsLoc 
               


#### Old stable step function before you modified numba balltree ####
def oldstbsteps(points,StabCrit):
    #construct search tree
    tree = sklearn.neighbors.BallTree(points,leaf_size=100)
    LevCts = np.empty(0)
    numActLevels = 35
    AppCount = np.zeros(35)
    Ct = 1
    while numActLevels > 1:
        print(Ct)
        seaR = np.sqrt(np.square(Ct) + np.square(Ct) + np.square(Ct))
        if Ct == 1:
            inds = tree.query_radius(points, seaR,return_distance=False)
        else:
            inds = tree.query_radius(EnPts, seaR,return_distance=False)

        fltinds = getfltinds(inds)

        if Ct == 1:
            pZlocs = points[fltinds[:,0],2]
            tZlocs = points[fltinds[:,1],2]
            possLocs = tZlocs>pZlocs
            possPtsLocs = fltinds[possLocs]
            StPtsLoc = np.setdiff1d(possPtsLocs[:,0],possPtsLocs[:,1])
            EnPtsLoc = np.setdiff1d(possPtsLocs[:,1],possPtsLocs[:,0])
            #also include in endpoints points that don't transition at all.
            AddEnPts = np.setdiff1d(np.arange(len(inds)),possPtsLocs.flatten())
            EnPtsLoc = np.union1d(EnPtsLoc,AddEnPts)
            EnPts = points[EnPtsLoc]
        else:
            pZlocs = EnPts[fltinds[:,0],2]
            tZlocs = points[fltinds[:,1],2]
            dropLocs = tZlocs>pZlocs
            dropPtsLocs = EnPtsLoc[fltinds[dropLocs,0]]
            EnPtsLoc = np.setdiff1d(EnPtsLoc,dropPtsLocs)
            EnPts = points[EnPtsLoc]

        #get number of levels
        Zlevs = EnPts[:,2]
        ActLevels = np.unique(Zlevs)
        print(ActLevels)
        #print(EnPts.shape)
        numActLevels = ActLevels.size
        LevCts = np.append(LevCts,[numActLevels])
        if Ct >= StabCrit:
            #stop increasing search distance if your number of levels has been stable for 'StabCrit' iterations
            if np.unique(LevCts[-StabCrit:]).size == 1 :
                #set_trace()
                #set the seaR you write to the sear at the base of your stabcrit (essentially correcting sear for repeat read criterion)
                seaR = np.sqrt(np.square(Ct-StabCrit+1) + np.square(Ct-StabCrit+1) + np.square(Ct-StabCrit+1))
                break

        Ct = Ct + 1

    return EnPts, numActLevels, seaR
    
def getfltinds(inds):
    li = list(map(len,inds))
    IniLocs = np.repeat(np.arange(len(inds)),li)
    TraLocs = np.concatenate(inds)
    FltInds = np.array([IniLocs,TraLocs]).T
    return FltInds
