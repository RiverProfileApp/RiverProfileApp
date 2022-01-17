## These are functions directly related to querying elevation and calulating chi

import numpy as np
from numba import jit
import math


@jit(nopython=True)
def lind(xy, n):
    """
    Compute linear index from 2 points

    :param xy: linear index
    :n: number of values in y dimension (or the number of values per row)
    """
    x = math.floor(int(xy) / n)
    y = xy % n
    return int(y), int(x)



@jit(nopython=True)
def calc_upstream(i, j, sx, sy, dem, acc, dx=90, athres=50, theta=.45):
    """
    takes the input flowdir indices sx sy and makes the topologically ordered
     stack of the stream network in O(n) time.  It also gathers elevation and topography data, and calculates chi based on the input theta value.
     The FastScape method does not properly order streams to be continuous, which is needed for plotting
      so stream indices are re-arranged after the fastscape method is employed.  Ideally the fastscape algorithm would be separated from these other
      calculations (so that we could have a separate (stack() function), but I could not think of an efficient way to decouple these other calculations
       from the stack calculation

    :param sx: x flow direction grid
    :param sy: y flow direction grid
    :param dem: input dem (mem-mapped)
    :param acc: drainage accumulation
    :param dx: x distance between cells
    :param athres: threshold for drainage area (minimum 10)
    :return: topologically ordered stack, I
    """

    dy = 92.6 # y distance is constant for hydrosheds
    athres *= dx * dy # Convert threshold area from pixels to meters
    ny, nx = np.shape(sx) #x and y dims
    sz = 5000000 #Initial pre-allocation size for arrays - we dont know the size of the basin a-priori

    #Pre allocating these is faster than adding to them progressively
    I = np.zeros(np.int64(sz), dtype=np.int64) # Stack linear indices
    dist = np.zeros(np.int64(sz), dtype=np.float64) - 1.0 # Cumulative distance up-stream
    z = np.zeros(np.int64(sz), dtype=np.float64) - 1.0 # Elevations
    chi = np.zeros(np.int64(sz), dtype=np.float64) # Chi values
    accid = np.zeros(np.int64(sz), dtype=np.int32) # River/ tributary id
    Av = np.zeros(np.int64(sz), dtype=np.float64) #Drainage area (vectorized)
    slps_basin = np.zeros(np.int64(sz), dtype=np.float64) #Slopes within the basin, for basin-wide metric calculation

    ij = j * ny + i # Linear index, iterates upstream

    i2 = i #The stream 2-d indices, these iterate upstream
    j2 = j

    c = 0
    k = 0
    c += 1

    I[c] = ij #Allocate the initial linear index, distance, elevation
    dist[c] = 0
    z[c] = dem[0, i, j]


    distl = 0.0 #The last downstream distance to be added to the next upstream value- for calculating cumulative distance
    accidl = 1 #The last downstream accumulation
    chil = 0 #The last downstream chi value

    cd = 0 # The id of current stream / tributary

    #Start with the initial point and find all its upstream donors, add to the stack
    while k < c < ny * nx - 1: #While there are still points on the stack ...

        maxa = 0 #Maximum drainage area will be determined from the loop
        maxcs = 0  #The index of maximum drainage area

        for i1 in range(-1, 2): # Determine if each neighbor of the current point is a donor
            for j1 in range(-1, 2):
                if 0 < j2 + j1 < nx - 1 and 0 < i2 + i1 < ny - 1:  # bounds check

                    ij2 = (j2 + j1) * ny + i2 + i1
                    recrx = sx[int(i2 + i1), int(j2 + j1)] #Get the receiver of the neighbors
                    recry = sy[int(i2 + i1), int(j2 + j1)]

                    if ((recrx != 0) or (recry != 0)) and ((recrx + j1 == 0) and (recry + i1 == 0)):#Is the current cell the receiver of an upstream cell?
                        # if current cell is a receiver, add it to the stack and add data to our arrays
                        A = np.float64(acc[i1 + i2, j1 + j2]) * dy * dx
                        I[c] = ij2
                        zi = dem[0, int(i2 + i1), int(j2 + j1)] #Elevation of donor
                        zr = dem[0, i2, j2] #Elevation of receiver
                        r = np.sqrt((j1 * dx) ** 2 + (i1 * dy) ** 2) #The distance between cells
                        slpi = (zi - zr) / r #Slope between top and its neighbor
                        s_pospart = (np.abs(slpi) + slpi) / 2 #We aren't interested in negative slopes
                        slps_basin[c] = s_pospart

                        if A >= athres:  # We only want to add streams above a given drainage area to the stream network
                            cd += 1
                            accid[c] = cd #Assign id to current stream

                            if A >= maxa: #The largest tributary of the current river keeps the ID of the downstream neighbor -  record which one that is
                                maxa = A
                                maxcs = c
                            dist[c] = distl + r
                            z[c] = dem[0, int(i2 + i1), int(j2 + j1)]
                            Av[c] = A
                            chi[c] = chil + A ** -theta * r

                        c += 1

        # cn = maxcs[p]

        if maxa > 0:
            accid[maxcs] = accidl

        k += 1
        ij = I[k]

        distl = dist[k] #To calculate cumulative distance, add to all the previous distance to all donor nodes
        accidl = accid[k]
        chil = chi[k]

        i2, j2 = lind(ij, ny) # Get the 2d indices of the next value on the stack


    dist = np.max(dist) - dist # Dist is to be plotted from the headwaters, not baselevel

    # Re-sort all streams so that they can be plotted linearly, i.e. the trunk stream, then tributary 1, tributary 2, etc.
    strm = np.zeros(0, dtype=np.int64)
    distn = np.zeros(0, dtype=np.float64)
    zn = np.zeros(0, dtype=np.float64)
    chin = np.zeros(0, dtype=np.float64)
    An = np.zeros(0, dtype=np.float64)


    id = np.logical_and(I > 0, accid > 0)  #Only want the indices of rivers (i.e., points on the stack above a certain threshold drainage area
    stats = {'na':0.00}
    bounds = 0

    if len(id[id]) > 0: # In case there are no streams found ...


        accid = accid[id] # Extract points in the stack that represent rivers, not hillslopes
        slps_riv = slps_basin[id]
        slps_basin = slps_basin[I > 0]
        I = I[id]
        z = z[id]
        dist = dist[id]
        chi = chi[id]
        Av = Av[id]

        area_basin = np.max(Av)
        slpmn = np.mean(slps_basin)
        slpstd = np.std(slps_basin)
        r_slpmn = np.mean(slps_riv)
        zmn = np.mean(z[z > 0])
        zstd = np.std(z[z > 0])

        # During sorting, add NaN at the end of the stream to mark the beginning of a new stream
        for i in range(1, np.max(accid)):
            idx = np.where(accid == i)[0]
            if len(idx) > 0:
                strm = np.append(np.append(strm, I[idx]), -9999)
                distn = np.append(np.append(distn, dist[idx]), np.nan)
                zn = np.append(np.append(zn, z[idx]), np.nan)
                chin = np.append(np.append(chin, chi[idx]), np.nan)
                An = np.append(np.append(An, Av[idx]), np.nan)
                # accidn = np.append(np.append(accid, accidn[idx]), np.nan)
        distn /= 1000
        stats = {"Basin Area (sq. km)": np.round(area_basin / 1e6, 2), "Avg Grad.": np.round(slpmn, 2),
                 "grade St. Dev.": np.round(slpstd, 2), "Avg River Grad.": np.round(r_slpmn, 2),
                 "Avg. Elev. (m)": np.round(zmn, 2),
                 "Elev. St. Dev. (m)": np.round(zstd, 2)}
        bounds = 0
    return strm, distn, zn, An, chin, stats, bounds


def chicalc(A, dist, theta, U=1):
    """
    Calculates chi values for linear stream (downstream mode).  Relatively trivial but used frequently, so makes sense
    to have a stand-alone function.

    :param A: Accumulation (linear) along stream
    :param dist: distance (linear) along stream
    :param theta: aka concavity
    :param U: If we have variable uplift
    :return: linear chi values along stream
    """
    chi = np.cumsum(U  * np.flip(A[:-1]) ** -theta * -np.diff(np.flip(dist))) * 1000 #(dist is in km )
    # chi[A<1e5] = np.nan
    return np.flip(chi)

@jit(nopython=True)
def getstream_elev(i, j, stackx, stacky, dem, elev=0):
    """
    Gets the indices from the receiver grids, until a given elevation
    (this is redundant, one of these should be replaced in the future)

    :param i: input initial index i
    :param j: input initial index j
    :param stackx: x receiver grid
    :param stacky: y receiver grid
    :param elev: elevation above which to profile to
    :return: linear index of stream locations on the grid
    """
    # nd=nd[0]
    ny, nx = np.shape(stackx)
    strm = np.zeros(int(ny * nx / 100), dtype=np.int64)
    z = np.zeros(int(ny * nx / 100), dtype=np.float64)
    c = 0
    while 1: # While there are downstream receivers
        i2 = i + np.int64(stacky[i, j])
        j2 = j + np.int64(stackx[i, j])
        i = i2
        j = j2
        ij = i + j * ny
        zi = dem[0, i, j]
        if zi < elev:
            break
        strm[c] = int(ij)
        z[c] = float(zi)
        c += 1
        if (stackx[i, j] == 0) and (stacky[i, j] == 0) or (i == ny - 2 or j == nx - 2 or j == 1 or i == 1):
            break
        if len(strm) > 1000000000000:
            break

    z = z[strm > 0]
    strm = strm[strm > 0]
    return strm, z


def find_nearest_thresarea(il, jl, acc, tol=10):
    """
    Finds the largest drainage area within a given box of specified km around the center at (il, jl) ("snapping").

    :param il: initial y coordinate
    :param jl: initial x coordinate
    :param acc: accumulation grid
    :param tol: tolerance (in units of cells) that defines the size of box
    :returns: coordinates of max. drainage area within the box
    """
    asub = acc[il - tol:il + tol,
           jl - tol:jl + tol]  # We have a 1 km square buffer to find the maximum drainage area near the clicked point
    ni, nj = np.unravel_index(np.argmax(asub), np.shape(asub))
    il += ni - tol
    jl += nj - tol
    return il, jl


