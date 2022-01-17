# coding=utf-8
from flask import Flask, render_template, request, Response, send_file, make_response, jsonify
from numba import jit
import numpy as np

import matplotlib
import io as IO
import base64
import geopandas as gpd

import pandas as pd
import altair as alt
from scipy.interpolate import interp1d as interpolate
# from basic_wrap import requires_auth
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from profiler_functions import chicalc, calc_upstream, getstream_elev, find_nearest_thresarea
from flask_caching import Cache
from shapely.geometry import LineString, Point, MultiLineString

#from waitress import serve

matplotlib.use('agg')

center = (-30, 130)  # where does initial load?
app = Flask(__name__)
config = {
    "DEBUG": False,  # some Flask specific configs
    "CACHE_TYPE": "filesystem",  # Flask-Caching related configs
    'CACHE_DIR': './templates/cache',
    "CACHE_DEFAULT_TIMEOUT": 300
}

# Data dirs for each continent
#basedir = '/media/data1/'
basedir = '/Volumes/t7/'

eudir = basedir + 'eu/'
nadir = basedir + 'na/'
sadir = basedir + 'sa/'
audir = basedir + 'au/'
afdir = basedir + 'af/'
marsdir = basedir + 'mars/'

# latlon ranges for each continent (top, left, bottom, right)
eurange = (60.0, -14.0, -10, 180)
sarange = (15.0, -93.0, -56, -32)
narange = (60.0, -145.0, 5.0, -52.00)
aurange = (-10.0, 112.0, -56.0, 180.0)
afrange = (40, -19.0, -35, 55)
marsrange = (90, -180, -90, 180)
ranges = [sarange, narange, aurange, afrange, eurange]  # Order matters, shouldn't be modified
dirs = [sadir, nadir, audir, afdir, eudir]

app.config.from_mapping(config)
cache = Cache(app)
res = 1200


def getz(dir1, ny, code, smooth=-1):
    """
    Get and store elevation values along stream.  If available, smoothed z values are automatically returned.  Otherwise, raw z values.
    If no raw z values are available, the function checks if the "strm" index variable has been set and collects z values directly from DEM.

    :param dir1:
    :param ny:
    :param code:
    :param smooth: smoothing wl in pixel
    :return: z values, smoothed
    """

    zfilt = cache.get('zfilt{}'.format(code))  # Does it already exist?
    suffx2 = cache.get('suffx2{}'.format(code))
    if zfilt is None:
        # If filtered z values don't exist already, does z exist atleast?
        zi = cache.get('z{}'.format(code))
        if zi is None:
            strm = cache.get('strm{}'.format(code))
            Is = strm % ny
            Js = np.int64(strm / ny)
            dem_s = np.load(dir1 + 'others/dem{}.npy'.format(suffx2), mmap_mode='r')
            zi = dem_s[0, Is, Js].copy()
            zi[zi < -30] = -30

        if smooth > 10:
            smooth = 10
        if smooth == -1:  # The default
            smooth = len(zi) / 500  # default

        # Otherwise we smooth using user prescribed value
        if smooth > 0:
            zfilt = gaussian_filter1d(zi, np.max([smooth, 1]))
        else:
            zfilt = zi.copy()
        if not (len(zfilt) == len(zi)):
            zfilt = zi.copy()
        cache.set('zfilt{}'.format(code), zfilt)
    return zfilt


def getA(dir1, ny, code, dx):
    """
    Get and cache the accumulation (linear) values along stream

    :param dir1: directory containing files
    :param ny: y dimensions
    :param code: code for caching
    :param dx: x resolution at this latitude
    :return: accumulation (linear) along stream
    """

    suffx = ''
    d8 = cache.get('d8{}'.format(code))

    if d8:
        suffx = '2'
    A = cache.get('acc{}'.format(code))
    if A is None:
        strm = cache.get('strm{}'.format(code))
        Is = strm % ny
        Js = np.int64(strm / ny)

        acc = np.load(dir1 + 'others/acc{}.npy'.format(suffx), mmap_mode='r')
        A = np.float64(acc[Is, Js] * 92.6 * dx)
        cache.set('acc{}'.format(code), A)
    return A


@app.route("/profiler")
# @requires_auth
def main():
    """
    Main page.  This is more or less a function for simple redirection to the main page.  But, it also initiates some important variables
    such as basin_extract (aka, upstream mode option) as well as d8 (unconditioned d8 option), and theta (for upstream mode chi plots).

    :return: rendered main page, no streams assigned yet (main page flag is 1)
    """
    code = np.random.randint(int(1e7))  # Identifier per user / better way too do this ???
    nuser = cache.get('nuser')
    basin_extract = request.args.get('basin_extract', type=int)
    d8 = request.args.get('d8', type=int)
    theta = request.args.get('theta', type=float)
    lon = request.args.get('longitude')
    lat = request.args.get('latitude')
    zoom = request.args.get('zoom')
    print(zoom)
    if lon is None:
        lon = 20
        lat = 43
    if theta is None:
        theta = 0.45
    if basin_extract is None:
        basin_extract = 0
    if zoom is None:
        zoom=5
    if d8 is None:
        d8 = 0
    if nuser is None:
        nuser = 0
    if nuser > 50:
        return "Sorry, too many users at the moment, try again in a few minutes"
    else:
        cache.set('nuser', nuser)
        return (
            render_template("main.html", data=str([[0, 0], [.1, .1]]), code=code, lat1=lat, lon1=lon, z=zoom, mainpage=1,
                            err='', zdata=str([[0, 0], [.1, .1]]), minz=0, maxz=0, dist='0', zfilt='0', n=1, maxdist=0,
                            elevl=-9999,
                            basin_extract=basin_extract, d8=d8, athres=1, theta=theta))


@app.route('/get_profiles')
# @requires_auth
def get_profiles():
    """
    Gets the location and elevation, drainage area, and other derivative information of stream profiles.
    This is the main driver function of RiverProfileApp.  Ultimately the all of the base functionaility is called from here, including
    downstream and upstream querying of rivers. The function assigns a code to a user which determines their "session"
    Importantly, several variables are saved in the cache by get_profiles which can then be used by subsequent calls to functions under a given session.
    These variables are

    acc: Drainage accumulation.  In upstream mode, individual rivers are separated by NaN
    dist: Distance downstream
    z: Raw elevation
    zfilt: Filtered elevation


    :return: elevations and locations of stream(s) rendered on main page
    """
    # Getter functions
    longitude = request.args.get('longitude', type=float)
    latitude = request.args.get('latitude', type=float)
    if longitude < -180:
        longitude += 360
    if longitude > 180:
        longitude -= 360

    code = np.random.randint(
        int(1e7))  # We don't expect many users at any given time, so this should be sufficiently random to generate a unique code

    zoom = request.args.get('zoom', type=int)
    d8 = request.args.get('d8', type=int)
    athres = request.args.get('athres', type=float)
    theta = request.args.get('theta', type=float)

    if athres is None or athres > 10 or athres < 0.1:
        athres = 1.0

    athres_pts = np.int(athres * 1e6 / 90 ** 2)
    basin_extract = request.args.get('basin_extract', type=int)
    dem2 = request.args.get('dem2', type=int)
    suffx2 = ''
    if dem2:
        suffx2 = '2'
    cache.set('suffx2{}'.format(code), suffx2)

    ## We want to restrict usage if too many queries
    nuser = cache.get('nuser')
    if nuser is None:
        nuser = 0
    nuser += 1
    cache.set('nuser', nuser)

    suffx = ''
    if d8:
        suffx = '2'
    elev = request.args.get('elev', type=int)
    smooth = request.args.get('smooth', type=float)

    # The directory dir1 depends on which DEM is being used
    dir1 = None
    for i in range(5):
        upper = ranges[i][0]
        lower = ranges[i][2]
        left = ranges[i][1]
        right = ranges[i][3]

        if i < 3:  # Every continent except for europe/ africa can be described as a box w / little overlap
            if left <= longitude < right:
                if lower <= latitude < upper:
                    dir1 = dirs[i]
                    ubound, lbound = (upper, left)
                    break
        elif i == 3:
            # We have to be careful about overlap between africa and europe 
            P = Point(longitude, latitude)
            g = gpd.read_file('af_bound')
            if g.contains(P)[0]:
                dir1 = afdir
                ubound, lbound = (upper, left)
                break
        elif i == 4:
            dir1 = eudir
            ubound, lbound = (upper, left)

    # Load the DEM and receiver grids using mmap
    dem_s = np.load(dir1 + 'others/dem{}.npy'.format(suffx2), mmap_mode='r')
    stackrx = np.load(dir1 + 'rs/stack_rx{}.npy'.format(suffx), mmap_mode='r')
    stackry = np.load(dir1 + 'rs/stack_ry{}.npy'.format(suffx), mmap_mode='r')
    acc = np.load(dir1 + 'others/acc{}.npy'.format(suffx), mmap_mode='r')

    #
    ny, nx = np.shape(stackrx)
    dy = 92.6
    dx = np.cos(np.abs(latitude) * np.pi / 180) * (1852 / 60) * 3  # Equation for dx depending on the latitude
    err = ''
    # Starting latitude and longitude
    il = int((ubound - latitude) * res)
    jl = int((longitude - lbound) * res)

    if basin_extract:
        tol = 10
        il, jl = find_nearest_thresarea(il, jl, acc, tol)
        if acc[il, jl] < 10000 * 1000000 / (dx * dy):  # Restrict possible drainage area
            strm, dist, z, A, chi, stats, bounds = calc_upstream(il, jl, stackrx, stackry, dem_s, acc, dx=dx, athres=athres_pts,
                                                         theta=theta)
            if len(strm) < 1:
                err += '| No streams were found with with high enough drainage area.  You may want to adjust the minimum drainage area or choose a new point (hint: zoom in to find a bigger stream)'
            else:
                if len(chi) > 0:
                    cache.set('chi{}'.format(code), chi)
                    cache.set('zfilt{}'.format(code), z)
                    cache.set('acc{}'.format(code), A)
        else:
            strm = np.zeros(1)
            dist = np.zeros(1)
            z = np.zeros(1)
            err += "| Selected drainage area is too large. Drainage area must be less than 10,000 sq. km . Try selecting a different point"

    else:
        strm, z = getstream_elev(il, jl, stackrx, stackry, dem_s, elev=elev)

    z[z < -30] = -30
    if len(strm) > 1 and len(err) == 0:
        if len(z) > 0:
            cache.set('z{}'.format(code), z)

        jo = np.float32(strm / ny) * 1 / res + lbound
        io = ubound - np.float32(strm % ny) * 1 / res

        id = np.where(strm < 0)[0]
        id = np.insert(id, 0, 0)

        if not (basin_extract):
            dist = np.cumsum(
                np.append(np.zeros(1), np.sqrt(((np.float64(io[:-1]) - np.float64(io[1:])) * dy * 1.2) ** 2 + (
                        (np.float64(jo[:-1]) - np.float64(jo[1:])) * dx * 1.2) ** 2)))

        if basin_extract and (len(id) > 0):

            maxdist = np.max(dist[~np.isnan(dist)])
            locations = [np.array(list(zip(io[id[i] + 1:id[i + 1]], jo[id[i] + 1:id[i + 1]]))).tolist() for i in
                         range(len(id) - 1)]


        else:
            locations = np.array(list(zip(io, jo))).tolist()
        data1 = locations
        ## Store data in the cache
        cache.set('dir{}'.format(code), dir1)
        cache.set('latitude{}'.format(code), latitude)
        cache.set('strm{}'.format(code), strm)
        cache.set('dist{}'.format(code), dist)
        cache.set('ny{}'.format(code), ny)
        cache.set('ubound{}'.format(code), ubound)
        cache.set('lbound{}'.format(code), lbound)
        cache.set('d8{}'.format(code), d8)
        cache.set('basin_extract{}'.format(code), basin_extract)

        if basin_extract:
            zfilt = z
        else:
            zfilt = getz(dir1, ny, code, smooth=smooth)
            stats = {"stats": 0}
        maxz = max(zfilt)
        # Interpolate to 1000 datapoints for the d3 plot
        distn = dist.copy()
        if not (basin_extract):
            dist = interpolate(np.arange(0, len(dist)), dist)(
                np.linspace(0.001, float(len(dist)) - 1.001, num=np.min([len(zfilt), 10000])))
            zfilt = interpolate(distn, zfilt)(dist)
            maxdist = np.max(dist[~np.isnan(dist)])
        dx = dist[1] - dist[0]

        # Requires list for d3 plot
        if basin_extract and (len(id) > 0):
            zdata = [np.array(list(zip(dist[id[i] + 1:id[i + 1]], zfilt[id[i] + 1:id[i + 1]]))).tolist() for i in
                     range(len(id) - 1)]
            distn = [np.array(list(zip(distn[id[i] + 1:id[i + 1]], distn[id[i] + 1:id[i + 1]]))).tolist() for i in
                     range(len(id) - 1)]
        else:
            zdata = np.array(list(zip(dist, zfilt))).tolist()
            distn = distn.tolist()
    else:  # If there's no stream there ...
        data1 = 0
        zdata = np.zeros(1)
        if len(strm) < 1:
            err += '| Sorry, no stream data found under the set parameters'
        return render_template("main.html", data=str(data1), code=code, lon1=longitude, lat1=latitude, z=zoom,
                               mainpage=1, err=err, maxz=0, maxdist=0, zdata=zdata, dist=0,
                               dx=0, elevl=-9999, basin_extract=basin_extract, d8=d8, theta=theta, stats=0, minz=0,
                               mindist=0, athres=athres)
    if not (basin_extract) and (smooth > 10):
        err += '| Smoothing was too high, reduced to 10'
    elif athres > 10 or athres < .1:
        err += '| Threshold area must be between 0.1 and 10 sq. km.  It has been automatically re-set to 1.0'
    # stats = pd.DataFrame(stats).to_html()
    return render_template("main.html", data=str(data1), code=code, lon1=longitude, lat1=latitude, z=zoom, mainpage=0,
                           err=err, maxz=maxz, minz=np.min(zfilt[zfilt > -10]), maxdist=maxdist, zdata=str(zdata),
                           dist=str(distn), dx=dx, elevl=elev, d8=d8, basin_extract=basin_extract, stats=stats,
                           theta=theta, athres=athres)


@app.route('/chiplot_multi')
# @requires_auth
def chiplot_multi():
    """
    Chi plots for upstream mode. Altair does not play well with the many chi-elevation plots, so this takes the
    chi-elevation data and converts it to a format which can then by processed by d3 in the chi_multi.html page.

    :return: chi - elevation interactive plot
    """
    ## Get the cached data and get variabless
    code = request.args.get('code', type=int)
    dir1 = cache.get('dir{}'.format(code))
    ny = cache.get('ny{}'.format(code))
    chi = cache.get('chi{}'.format(code))
    zfilt = getz(dir1, ny, code)
    # Distance and z values must be interpolated
    id = np.where(np.isnan(zfilt))[0]
    id = np.insert(id, 0, 0)
    maxz = np.max(zfilt[~np.isnan(zfilt)])
    minz = np.min(zfilt[~np.isnan(zfilt)])
    maxdist = np.max(chi[~np.isnan(chi)])
    data = [np.array(list(zip(chi[id[i] + 1:id[i + 1]], zfilt[id[i] + 1:id[i + 1]]))).tolist() for i in
            range(len(id) - 1)]

    return render_template("chi_multi.html", data=data, minz=minz, maxz=maxz, maxdist=maxdist)  # ,im = io1)


@app.route('/chiplot')
# @requires_auth
def chiplot():
    """
    This main chi plot is done primarily by Altair.  This calculates chi from acc and dist vectors,
    converts chi - elevation data into a Pandas dataframe, and returns it as html script via the to_html function built
    into Altair

    :return: chi - elevation interactive plot
    """
    ## Get the cached data and get variabless
    code = request.args.get('code', type=int)
    dir1 = cache.get('dir{}'.format(code))
    latitude = cache.get('latitude{}'.format(code))
    ny = cache.get('ny{}'.format(code))
    dx = np.cos(np.abs(latitude) * np.pi / 180) * (1852 / 60) * 3
    dist = cache.get('dist{}'.format(code))
    zfilt = getz(dir1, ny, code)
    d8 = cache.get('d8{}'.format(code))
    A = getA(dir1, ny, code, dx)

    # Distance and z values must be interpolated
    distn = interpolate(np.arange(1, len(dist) + 1), dist)(np.linspace(1, float(len(dist)), num=1000))
    zfiltn = interpolate(dist, zfilt)(distn)
    An = interpolate(dist, A)(distn)
    chi = chicalc(A, dist, .45, U=1.0)
    ## Calculate and interpolate chi for various values of theta to be compared
    p = pd.DataFrame()
    for theta in np.arange(.05, 1.05, .1):
        chi2 = chicalc(A, dist, theta, U=1.0)

        chi1 = interpolate(np.arange(0, len(chi2)), chi2)(np.linspace(0, float(len(chi2)) - 1.01, num=1000))
        zfiltn = interpolate(chi2, zfilt[:-1])(chi1)
        p2 = pd.DataFrame({'Z': zfiltn, 'χ': chi1, 'θ': np.round(np.zeros(len(chi1)) + theta, 2)})
        p = p.append(p2)

    plt.xlabel('chi (m)')
    plt.ylabel('Elevation (m)')

    cache.set('chi{}'.format(code), chi)

    # Altair plot
    alt.data_transformers.enable('default', max_rows=1000000)
    brush = alt.selection_interval(encodings=['x'])
    slider = alt.binding_range(min=.05, max=.95, step=.1)
    select_theta = alt.selection_single(name="theta value", fields=['θ'],
                                        bind=slider, init={'θ': .45})

    chart1 = alt.Chart().mark_line().encode(
        x=alt.X('χ:Q', scale=alt.Scale(zero=False)),
        y=alt.Y('Z:Q', scale=alt.Scale(zero=False)),
    ).properties(
        width=600,
        height=500
    ).add_selection(brush
                    ).add_selection(select_theta
                                    ).transform_filter(select_theta)
    chart2 = alt.Chart().mark_line().encode(
        x=alt.X('χ:Q', scale=alt.Scale(zero=False)),
        y=alt.Y('Z:Q', scale=alt.Scale(zero=False)),
    ).properties(
        width=600,
        height=500
    ).transform_filter(
        brush).transform_filter(select_theta)
    chart = alt.vconcat(
        chart1,
        chart2,
        data=p,
        title="Select data to analyze in the top panel, use the slider at the bottom to modify theta"
    )

    io2 = chart.to_html()

    # Just return the altair plot as html code
    return render_template("im.html", code=code, data=io2)  # ,im = io1)


@app.route('/other_functions')
def other_functions():
    """
    This routes to the other_functions html page for convenience

    :returns: other_functions.html, which is intentionally simplified.  Only the user code is passed
    """
    code = request.args.get('code', type=int)
    return render_template("other_functions.html", code=code)

@app.route('/get_elev')
# @requires_auth
def get_elev():
    """
    Getter for elevation data.  This is done by a call to getz(), which returns smoothed data (in downstream mode) if available.
    otherwise, it returns the raw data.


    :return:     Get elevation to download
    """
    code = request.args.get('code', type=int)
    dir1 = cache.get('dir{}'.format(code))
    ny = cache.get('ny{}'.format(code))
    zfilt = getz(dir1, ny, code)
    data1 = zfilt.tolist()

    if len(data1) == 0:
        return render_template("download.html", data='No Data')
    return render_template("download.html", data=str(data1).replace('[', '').replace(']', ''), name='elev', code=code,
                           json=0)


@app.route('/get_dist')
# @requires_auth
def get_dist():
    """
    Download distance array
    Please note units for dist used to be degrees in previous versions of the app, changed to the more standard (meters)

    :return: Distance to download 
    """
    code = request.args.get('code', type=int)
    dist = cache.get('dist{}'.format(code))
    if dist is None:
        return render_template("download.html", data='No Data')
    data1 = dist.tolist()
    return render_template("download.html", data=str(data1).replace('[', '').replace(']', ''), name='dist', code=code,
                           json=0)


@app.route('/get_acc')
# @requires_auth
def get_acc():
    """
    Get the accumulation grids to download - just a simple call to getA.  This can be done by a direct call to the cache also

    :return: Acc to download
    """
    code = request.args.get('code', type=int)

    latitude = cache.get('latitude{}'.format(code))

    dir1 = cache.get('dir{}'.format(code))
    ny = cache.get('ny{}'.format(code))
    dx = np.cos(np.abs(latitude) * np.pi / 180) * (1852 / 60) * 3
    d8 = cache.get('d8{}'.format(code))

    A = getA(dir1, ny, code, dx)
    if A is None:
        return render_template("download.html", data=str('No Data'))
    data1 = A.tolist()
    return render_template("download.html", data=str(data1).replace('[', '').replace(']', ''), name='acc', code=code,
                           json=0)


@app.route('/get_chi')
# @requires_auth
def get_chi():
    """
    This will get the chi values only if a chi plot has already been generated (in downstram mode).  A chi plot does not need to be
    generated in upstream mode.

    :return: chi values to download
    """
    code = request.args.get('code', type=int)

    chi = cache.get('chi{}'.format(code))

    if chi is None:
        return "No Data yet - go back and generate a chi plot first!"
    data1 = chi.tolist()
    return render_template("download.html", data=str(data1).replace('[', '').replace(']', ''), name='χ', code=code,
                           json=0)


@app.route('/get_shp')
# @requires_auth
def get_shp():
    """
    Generates a shapefile from the river network.  This uses basic functionality of geopandas.  In reality it returns it in
    geojson format, but due to lack of familiarity with that format the name of the function reflects shapefile, which more people
    are probably familiar with

    :return: geojson - a bit misleading but name is descriptive
    """
    code = request.args.get('code', type=int)
    strm = cache.get('strm{}'.format(code))
    ny = cache.get('ny{}'.format(code))
    lbound = cache.get('lbound{}'.format(code))
    ubound = cache.get('ubound{}'.format(code))
    basin_extract = cache.get('basin_extract{}'.format(code))
    jo = np.float64(strm / np.float64(ny)) * 1.0 / res + lbound
    io = ubound - np.float64(strm % ny) * 1.0 / res
    id = np.where(strm < 0)[0]
    id = np.insert(id, 0, 0)
    locations = []
    if (basin_extract):
        for i in range(len(id) - 1):
            ij = np.array(list(zip(jo[id[i] + 1:id[i + 1]], io[id[i] + 1:id[i + 1]]))).tolist()
            if len(ij) > 1:
                locations.append(ij)
        L = MultiLineString(locations)  # for IJ in locations]
    else:
        IJ = list(zip(jo, io))
        L = LineString(IJ)
    g = gpd.GeoDataFrame(geometry=[L])
    data1 = g.to_json()
    return Response(data1,
                    mimetype='application/json',
                    headers={'Content-Disposition': 'attachment;filename={}.geojson'.format(code)})


## This is an example, but provides a robust framework for adding additional functionaliry
@app.route('/boxplots')
def boxplots():
    """
    This is an example code to show how to take in variables generated by the app during upstream and downstream querying
    and then applying a custom function, in this case returning a box plot of elevations sorted by drinage areas as a png figure

    :Returns: boxplot figure, binned acc vs elevation
    """
    # First get the code for the user's session
    code = request.args.get('code', type=int)

    #Now get some important variables - these are a few of them, you can find examples thoughout the app the others that are stored such as linear indices of the stack ( strm{}.format(code))
    dist = cache.get('dist{}'.format(code))
    z = cache.get('zfilt{}'.format(code))
    acc = cache.get('acc{}'.format(code))

    acc = acc[z>0]
    dist = dist[z>0]
    z = z[z>0]

    #Just create a simple boxplot image of binned drainage area vs elevation
    nbins = 7
    p = pd.DataFrame({'z (m)': z,'A':acc,'D':dist}, index = range(len(z[np.logical_not(np.isnan(z))])))
    bins = pd.qcut(p['A'],nbins)
    p['Average Area (m^2)'] = [round(bins[i].left/2 + bins[i].right/2) for i in range(len(bins))]

    plt.figure(figsize=(16,8))
    sns.boxplot(data = p, x='Average Area (m^2)', y='z (m)')

    #This saves the image to a place in memory rather than a file
    F = IO.BytesIO()
    plt.savefig(F)
    F.seek(0) # This line is required, for some reason ...

    #This will display whatever you send it via the chart variable  - can be text or table or an image or json or html code.
    return send_file(F, mimetype='image/png')


if __name__ == "__main__":
    app.run()  # host='0.0.0.0', port=13211)
    #serve(app, host='0.0.0.0', port=13211)
