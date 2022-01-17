
# ProfilerApp

## Seting up your own implementation of the server
Global stream profiler web app. 

To set up, add the appropriate directories in, global variables in profiler_app.py, eg:

basedir = '/media/data1/' 

eudir = basedir + 'eu/'


If you are running a small server, you can get away with Flask's built-in caching.  However, a better caching service such as Waitress is probably necessary for larger-scale implementations.  An example using waitress is commented out

Tests can be run on both the server and to verify integrity of the data, these are available with instructions from the notebooks in the tests folder 

You can obtain the grid files from the google drive link, and unzip them:

https://drive.google.com/drive/folders/144QhSimTndRFO2n0Cn9y4gcwE_zzBPKN?usp=sharing

Or, for power users, you can build the grids yourself using the functions in the GlobalStack repository (see more details in the readme file there).  The app currently assumes 3 arc-second resolution of the grids.  You will need to change the "res" global variable if your grids are different (res is currently set to be the values of pixels/degree, so for 3 arc-seconds this is 1200)

## Custom functions

Community contributions are definitely welcome. However at the moment, I am trying not to crowd the main webpage with too many functions. So I have created a new page called "other_functions" where links to all future functions can be easily added.  
If you are interested in adding custom functions, the best way to do that would be to send me the function directly.  I can evaluate it, decide whether it fits with the current overall goals of the app, and add then it into the interface.

Alternatively, you can fork your own version and add your own functionality directly.  I added an example function called "boxplots" within the profiler_app.py module that shows how to do this.  Basically, any function that uses the elevation data, drainage area data, or other metrics derived from these (such as slope, curvature etc.) can be directly implemented.  A "session" is started once the user clicks on the map and hits submit.  This initiates the "get_profiles" function, which finds all streams upstream or downstream of the selected point, depending on the mode the user is selecting their streams from.  The "get_profiles" function then saves drainage area, elevation, and location of streams within the server cache.  Subsequent calls to other functions can then use these variables for their own calculations.  

For example, the "boxplots" function is called once a user selects and requests stream data, then clicks on the link to the "boxplots" function. The boxplots function can then gather the saved data from the cache, and use it for its own calculations.  In this case, boxplots creates a figure of binned drainage area vs elevation, and sends the figure directly to the user to display or download.  

The following variables can be called from the cache after running get_profiles:


**strm**: Linear Indices of the river on the current grid

**ny**: Number of rows of the grid.  I and J (row and column) bilinear indices of the grid can be calculated by I = strm%ny and J = int(strm/ny).  

**d8** The user option, whether or not to use the unfiltered d8 grids

**basin_extract**: Are we in upstream mode?

**acc**: Drainage accumulation of the streams.  In upstream mode, individual rivers are separated by NaN

**dist**: Distance downstream

**z:** Raw elevation

**zfilt**: Filtered (by gaussian filter) elevation

**dir**: Location of DEM, receiver grids, etc. in the server's file system 

**chi**: Chi values, which are calculated ahead during call to get_upstream in upstream mode (in downstream mode, the calculation is fairly trivial and thus it is recommended to be done as a function of acc and dist during runtime)

**latitude / longitude** : The user's selected latitude and longitude 

A variable is called by using the cache.get function with the session code appended to the variable name. 
For example, to get the accumulation data after call the function from the html page 
while passing the user's code to the function from the webpage (see other_functions.html for an example on how to create a link) ,  do the following:

code = request.args.get('code', type=int)

acc = cache.get('acc{}'.format(code))

## Use custom functions for other purposes

All of the functions in profiler_app.py contain what I consider to be baseline functionality that are not necessarily specific to the RiverProfileApp.  Any function, particularly those such as get_upstream can be used as part of other projects.  Also consider additional functionality in the GlobalStack repository.

## Documentation

Sphinx documentation on all functions can be found at:  riverprofileapp.github.io/sphinx_index.html