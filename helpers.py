import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial.distance import pdist, squareform
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import normalize
from scipy.spatial.distance import squareform,pdist
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
import csv
import  pandas  as pd
from  sklearn.cluster  import  DBSCAN
from  geopy.distance  import  great_circle
from  shapely.geometry  import  MultiPoint
import matplotlib.pyplot as plt
from time import time
import io
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding
from haversine import haversine
from vincenty import vincenty
import warnings; warnings.simplefilter('ignore')


def fix_table(A):
    for i in range(len(A)):
        for j in range(len(A)):
                A[i,j]=A[j,i]
                #print(i,j,A[i,j],A[j,i])
def check_sym(A):
    if check_symmetric(A) is False:
        fix_table(A)
def euclidean_distances2(A):
    kart=euclidean_distances(A)
    check_sym(kart)
    return kart
def fix_table2(A):
    for i in range(len(A)):
        for j in range(len(A)):
            d=int((A[i,j]-A[j,i])*10000)/1000
            if d!=0:
                #print(i,j,A[i,j],A[j,i])
                A[i,j]=A[j,i]
                #print(i,j,A[i,j],A[j,i])
    
def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)
def myplotM(mylats1,mylons1):    
    #%matplotlib qt
    zoom_scale = 0

    # Setup the bounding box for the zoom and bounds of the map
    bbox = [np.min(mylats1)-zoom_scale,np.max(mylats1)+zoom_scale,\
            np.min(mylons1)-zoom_scale,np.max(mylons1)+zoom_scale]

    plt.figure(figsize=(12,6))
    # Define the projection, scale, the corners of the map, and the resolution.
    m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
                llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')

    # Draw coastlines and fill continents and water with color
    m.drawcoastlines()
    m.fillcontinents(color='blue',lake_color='dodgerblue')

    # draw parallels, meridians, and color boundaries
    m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
    m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=45)
    m.drawmapboundary(fill_color='dodgerblue')

    # build and plot coordinates onto map
    x,y = m(mylons1,mylats1)
    xMerc=x
    yMerx=y
    m.plot(x,y,'r*',markersize=3)
    plt.title("Pre clustered data")
    plt.savefig('gr_sub.png', format='png', dpi=500)
    plt.show()
def p2cart(mylats,mylons):
    fdata21  =np.array(lon_lat_to_cartesian2(mylats,mylons),dtype=np.float32)
    fdata21=np.transpose(fdata21)
    return fdata21
def lon_lat_to_cartesian2(lat, lon):
    import numpy as np
    import numpy as Math
    import math 
    pi=math.pi
    
    lon=np.array(lon,dtype=np.float32)
    lat=np.array(lat,dtype=np.float32)
    
    cosLat = Math.cos(lat * pi / 180.0)
    sinLat = Math.sin(lat * pi / 180.0)
    cosLon = Math.cos(lon * pi / 180.0)
    sinLon = Math.sin(lon * pi / 180.0)
    
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon=np.array(lon,dtype=np.float32)
    lat=np.array(lat,dtype=np.float32)
    rad = 6378137.0
    f = 1.0 / 298.257224
    C = 1.0 / Math.sqrt(cosLat * cosLat + (1 - f) * (1 - f) * sinLat * sinLat)
    S = (1.0 - f) * (1.0 - f) * C
    h = 0.0;
    x = (rad * C + h) * cosLat * cosLon
    y = (rad * C + h) * cosLat * sinLon
    z = (rad * S + h) * sinLat
    #print(x,y,z)
    return [x,y,z]
def cart_to_ll(xy):
    R1 = 6378100
    xy=np.array(xy,dtype=np.float32)
    x=xy[:,1]
    y=xy[:,0]
    #(x, y, z) = xy # does not use `z`
    r = np.sqrt(x**2 + y**2)/R1
    long =np.array(180 * np.arctan2(y,x)/math.pi,dtype=np.float32)
    lat = 180 *np.arccos(r)/math.pi
    return np.transpose(np.array((lat,long),dtype=np.float32))
def Merc(mylats1, mylons1):
    import numpy as np
    import numpy as Math
    import math 
    pi=math.pi
    zoom_scale = 0
     
    
    
    mylons1=np.array(mylons1,dtype=np.float32)
    mylats1=np.array(mylats1,dtype=np.float32)
    
    bbox = [np.min(mylats1)-zoom_scale,np.max(mylats1)+zoom_scale,\
            np.min(mylons1)-zoom_scale,np.max(mylons1)+zoom_scale]

    #plt.figure(figsize=(12,6))
    # Define the projection, scale, the corners of the map, and the resolution.
    m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
                llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')

    # Draw coastlines and fill continents and water with color
    m.drawcoastlines()
    m.fillcontinents(color='blue',lake_color='dodgerblue')

    # draw parallels, meridians, and color boundaries
    m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
    m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=45)
    m.drawmapboundary(fill_color='dodgerblue')

    # build and plot coordinates onto map
    x,y = m(mylons1,mylats1)
    xMerc=x
    yMerx=y
    #print(xMerc,yMerx)
    return [xMerc,yMerx]
def haversine2(u,v,d):
    dis=haversine(u,v)
    if dis<=d:
        return dis
    else:
        return 0
def fill_color(X,d=3):
    import collections
    import matplotlib.colors as mcolors
    colors2=mcolors.CSS4_COLORS
    by_hsv = ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                        for name, color in colors2.items())

    by_hsv = sorted(by_hsv)
    cnames = [name for hsv, name in by_hsv]
    cnm=len(cnames)
    color=[]
    if d==3:
        cco=list(collections.Counter(X[:,d]))
        len(cco)

        cld={}
        
        for cf in cco:
            if cf=='Africa':
                n='lime'
            if cf=='America':
                n='red'
            if cf=='Arctic':
                n='gray'
            if cf=='Asia':
                n='gold'
            if cf=='Atlantic':
                n='cyan'
            if cf=='Australia':
                n='g'
              
            if cf=='Europe':
                n='b'
            if cf=='Indian':
                n='peru'
            if cf=='Pacific':
                n='c'
                
            cld[cf]=n
            
        

        for l in X[:,d]:
            color.append(cld[l])
        
    else:
        color=[]
        cco=list(collections.Counter(X[:,d]))
        len(cco)
        
        cld={}
        n=1
        for cf in cco:
            
            cld[cf]=cnames[n]
            n+=2
            if n>=cnm:
                n=1

        for l in X[:,d]:
            color.append(cld[l])
    return color

# {'Africa',
#  'America',
#  'Arctic',
#  'Asia',
#  'Atlantic',
#  'Australia',
#  'Europe',
#  'Indian',
#  'Pacific'}


def visualize3DData (X,title,color):
    """Visualize data in 3d plot with popover next to mouse position.

    Args:
        X (np.array) - array of points, of shape (numPoints, 3)
    Returns:
        None
    """
   
    fig = plt.figure(figsize = (16,10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X[:, 6], X[:, 7], X[:, 8], depthshade = False, c=color, picker = True,s=10)
    plt.gcf().canvas.set_window_title(title)

    def distance(p1,p2,p3, event):
        """Return distance between mouse position and given data point

        Args:
            point (np.array): np.array of shape (3,), with x,y,z in data coords
            event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64): distance (in screen coords) between mouse pos and data point
        """
        point=np.array([p1,p2,p3])
        assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)


    def calcClosestDatapoint(X, event):
        """"Calculate which data point is closest to the mouse position.

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        distances = [distance (X[i, 6],X[i, 7],0, event) for i in range(X.shape[0])]
        return np.argmin(distances)


    def annotatePlot(X, index):
        """Create popover label in 3d chart

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
        Returns:
            None
        """
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2, y2, z2 = proj3d.proj_transform(X[index, 6], X[index, 7], X[index, 8], ax.get_proj())
        annotatePlot.label = plt.annotate( "{}:\n{},{}\n{},{}\n{} {}".format(X[index, 2],X[index, 0], X[index, 1],X[index, 6], X[index, 7], X[index, 3]+'-'+X[index, 5], X[index, 4]) ,
            xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.8),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        fig.canvas.draw()

def visualize2DData (X,title,d=3,col='midnight'):
    """Visualize data in 3d plot with popover next to mouse position.

    Args:
        X (np.array) - array of points, of shape (numPoints, 3)
    Returns:
        None
    """
    
    color=fill_color(X,d)
    
#     import collections


#     cco=list(collections.Counter(X[:,d]))
#     len(cco)

#     cld={}
#     n=1
#     for cf in cco:
#         cld[cf]=n
#         n+=200
#     color=[]
#     for l in X[:,d]:
#         color.append(cld[l])
    fig = plt.figure(figsize = (16,10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X[:, 6], X[:, 7], X[:, 6]*0, depthshade = False, c=color, picker = True,s=10)
    ax.set_facecolor('xkcd:'+col)
    plt.gcf().canvas.set_window_title(title)

    def distance(p1,p2,p3, event):
        """Return distance between mouse position and given data point

        Args:
            point (np.array): np.array of shape (3,), with x,y,z in data coords
            event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64): distance (in screen coords) between mouse pos and data point
        """
        point=np.array([p1,p2,p3])
        assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)


    def calcClosestDatapoint(X, event):
        """"Calculate which data point is closest to the mouse position.

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        distances = [distance (X[i, 6],X[i, 7],0, event) for i in range(X.shape[0])]
        return np.argmin(distances)


    def annotatePlot(X, index):
        """Create popover label in 3d chart

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
        Returns:
            None
        """
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2, y2, _ = proj3d.proj_transform(X[index, 6], X[index, 7], X[index, 6]*0, ax.get_proj())
        annotatePlot.label = plt.annotate( "{}:\n{},{}\n{},{}\n{} {}".format(X[index, 2],X[index, 0], X[index, 1],X[index, 6], X[index, 7], X[index, 3]+'-'+X[index, 5], X[index, 4]) ,
            xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.8),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        fig.canvas.draw()


    def onMouseMotion(event):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
        closestIndex = calcClosestDatapoint(X, event)
        annotatePlot (X, closestIndex)

    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
    plt.show()
def find_error(p1,p2,X,debug):
    from scipy.spatial import distance
    import math
       
    fd1=euclidean_distances([X[p1,5:7],X[p2,5:7]])[0,1]  
    d1=euclidean_distances([X[p1,0:2],X[p2,0:2]])[0,1]
                 
    
    Dd=abs(fd1-d1)
    if debug:
        print("Start:",X[p1,3],X[p1,2])
        print(X[p1,0],",",X[p1,1])
        print("End:",X[p2,3],X[p2,2])
        print(X[p2,0],",",X[p2,1])
        print("dist:",fd1,",",d1)
        print(X[p1,5],X[p1,6],X[p2,5],X[p2,6])
        print(X[p1,0],X[p1,1],X[p2,0],X[p2,1])
    return (fd1-d1)**2,d1**2
def check_result(cou,P,debug):
    a=0
    b=0
    asum=0
    bsum=0
    from random import randint

    for i in range(0,len(P)):
        #print("step:",i)
        for j in range(0,len(P)):
                a,b=find_error(i,j,P,debug)
                asum+=a
                bsum+=b
    print("Stress ",math.sqrt(asum/bsum))
def  get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    # print("centroid",centroid)
    # print("centroid2",MultiPoint(cluster).centroid)
    centermost_point = min(cluster , key=lambda  point: great_circle(point , centroid).m)
    return  tuple(centermost_point)


def ccluster(coor,ep):  
    print ("ccluster",len(coor))
    kms_per_radian = 6371.0088
    epsilon = ep/ kms_per_radian
    db = DBSCAN(eps=epsilon ,
    min_samples =1,
    algorithm='ball_tree',
    metric='haversine').fit(np.radians(coor[:,0:2]))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series ([ coor[cluster_labels  == n] for n in  range(num_clusters)])
    print('Number  of  clusters: {}'.format(num_clusters))
    #print(clusters[0:3])
    centermost_points = clusters.map(get_centermost_point)
    #print ("cpoints",centermost_points)
    return centermost_points
def start_cluster(coo ,st,minp,df):    

    cpoints=ccluster(coo,st)
    print(len(cpoints),st)
    lats , lons ,geoid = zip(* cpoints)
    rep_point = pd.DataFrame ({  'lat':lats,'lon':lons, 'geoid':geoid})
    print(len(rep_point),st)
    mycities2=rep_point.values

    if minp>0:

        cpoints=ccluster(mycities2,2*st)
        print(len(cpoints),2*st)
        lats , lons ,geoid = zip(* cpoints)
        rep_point = pd.DataFrame ({  'lat':lats,'lon':lons, 'geoid':geoid})
    
    if minp>0:

        if len(cpoints)>minp:
            mycities2=rep_point.values
            cpoints=ccluster(mycities2,4*st)
            print(len(cpoints),4*st)
            lats , lons ,geoid = zip(* cpoints)
            rep_point = pd.DataFrame ({  'lat':lats,'lon':lons, 'geoid':geoid})
    
        if len(cpoints)>minp:
            mycities2=rep_point.values
            cpoints=ccluster(mycities2,6*st)
            print(len(cpoints),6*st)
            lats , lons ,geoid = zip(* cpoints)
            rep_point = pd.DataFrame ({  'lat':lats,'lon':lons, 'geoid':geoid})
        
        if len(cpoints)>minp:
            mycities2=rep_point.values
            cpoints=ccluster(mycities2,8*st)
            print(len(cpoints),8*st)
            lats , lons ,geoid = zip(* cpoints)
            rep_point = pd.DataFrame ({  'lat':lats,'lon':lons, 'geoid':geoid})
    
        if len(cpoints)>minp:
            mycities2=rep_point.values
            cpoints=ccluster(mycities2,10*st)
            print(len(cpoints),10*st)
            lats , lons ,geoid = zip(* cpoints)
            rep_point = pd.DataFrame ({  'lat':lats,'lon':lons, 'geoid':geoid})
    
        if len(cpoints)>minp:
            mycities2=rep_point.values
            cpoints=ccluster(mycities2,12*st)
            print(len(cpoints),12*st)
            lats , lons ,geoid = zip(* cpoints)
        
        if len(cpoints)>minp:
            mycities2=rep_point.values
            cpoints=ccluster(mycities2,15*st)
            print(len(cpoints),15*st)
            lats , lons ,geoid = zip(* cpoints)
        
    rnum=len(geoid)
    rr=([df[df[0]==geoid[n]].as_matrix(columns =[2,17,8,18,19,20]) for n in  range(len(geoid))])
    rr1=[]
    rr3=[]
    rr4=[]
    rr5=[]
    rr6=[]
    rr7=[]
    
    for  o in rr:
        rr1.append(o[0][0])
        rr3.append(o[0][1])
        rr4.append(o[0][2])
        rr5.append(o[0][3])
        rr6.append(o[0][4])
        rr7.append(o[0][5])
    #rr2=pd.Series([df[df[0]==geoid[n]][8] for n in  range(len(geoid))])
    #rr2=([df[df[0]==geoid[n]].as_matrix(columns =[8])[0] for n in  range(len(geoid))]) 
    #rr2=rr2.iloc[:]
       
         
    rep_point = pd.DataFrame ({  'lat':lats,'lon':lons, 'name':rr1,'country':rr3, 'geoid':geoid,'cont':rr4, 'x': rr5, 'y':rr6, 'z':rr7})
    mycities2=rep_point.values
    return rep_point
def myplot(mylats1,mylons1,myz=0): 
 
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(mylats1, mylons1, myz)
    pyplot.show()
def plotmap(coords,mtitle):
    lats=coords[:,0]
    lons=coords[:,1]
    gid=coords[:,2]
    #coords = df.as_matrix(columns =['LAT','LON','ELEV'])        
    # How much to zoom from coordinates (in degrees)
    zoom_scale = 0
    # Setup the bounding box for the zoom and bounds of the map
    bbox = [np.min(lats)-zoom_scale,np.max(lats)+zoom_scale,\
            np.min(lons)-zoom_scale,np.max(lons)+zoom_scale]
    plt.figure(figsize=(12,6))
    # Define the projection, scale, the corners of the map, and the resolution.
    m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
                llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')
    # Draw coastlines and fill continents and water with color
    m.drawcoastlines()
    m.fillcontinents(color='blue',lake_color='dodgerblue')
    # draw parallels, meridians, and color boundaries
    m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
    m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=45)
    m.drawmapboundary(fill_color='dodgerblue')
    # build and plot coordinates onto map
    x,y = m(lons,lats)
    m.plot(x,y,'r*',markersize=3)
    xMerc=np.array(x,dtype=np.float32)
    yMerc=np.array(y,dtype=np.float32)
    xyMerc=np.array([xMerc,yMerc],dtype=np.float32)
    plt.title(mtitle)
    plt.savefig('mtitle', format='png', dpi=500)
    plt.show()
    return xyMerc
def plotmap2(coords,mtitle):
    lats=coords[:,0]
    lons=coords[:,1]
    gid=coords[:,2]
    #coords = df.as_matrix(columns =['LAT','LON','ELEV'])        
    # How much to zoom from coordinates (in degrees)
    zoom_scale = 0
    # Setup the bounding box for the zoom and bounds of the map
    bbox = [np.min(lats)-zoom_scale,np.max(lats)+zoom_scale,\
            np.min(lons)-zoom_scale,np.max(lons)+zoom_scale]
    #plt.figure(figsize=(12,6))
    # Define the projection, scale, the corners of the map, and the resolution.
    m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
                llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')
    # Draw coastlines and fill continents and water with color
    #m.drawcoastlines()
    #m.fillcontinents(color='blue',lake_color='dodgerblue')
    # draw parallels, meridians, and color boundaries
    #m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
    #m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=45)
    #m.drawmapboundary(fill_color='dodgerblue')
    # build and plot coordinates onto map
    x,y = m(lons,lats)
    #m.plot(x,y,'r*',markersize=3)
    xMerc=np.array(x,dtype=np.float32)
    yMerc=np.array(y,dtype=np.float32)
    xyMerc=np.array([xMerc,yMerc],dtype=np.float32)
    #plt.title(mtitle)
    #plt.savefig('mtitle', format='png', dpi=500)
    #plt.show()
    m=None
    return xyMerc

def cont(country):
    country=[num for num in countries if num['code'] ==country]
    return country[0]['continent']

def merc_ll(lats, lons):
    r_major = 6378137.000
    x = r_major * np.radians(lons)
    scale = x/lons
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lats * (np.pi/180.0)/2.0)) * scale
    return np.array(x, y)


    
def doMDS(fdata,fdataM,xx):
    import numpy as np
    t0=time()
    seed = np.random.seed(seed=3)
    #seed=1
    #CALCULATE DIF MATRICES

    a=squareform(pdist(fdata, (lambda u,v: haversine(u,v))))
#     distance_matrixX  = squareform(pdist(xx,'sqeuclidean'))
#     distance_matrix2  = squareform(pdist(fdataM,'sqeuclidean'))
#     distance_matrix7  = squareform(pdist(gvdata,'sqeuclidean'))
#     distance_matrix = squareform(pdist(fdata2,'sqeuclidean'))
    #distance_matrixc = squareform(pdist(fdatac,'sqeuclidean'))
    distance_matrixc2  = euclidean_distances(fdataM.astype(np.float64))


    mds7 = MDS(2,max_iter=100, 
                       dissimilarity="precomputed", n_jobs=-1)
    temp7=mds7.fit(distance_matrixc2)
    stress7=temp7.stress_
    tdata7 = mds7.fit_transform(distance_matrixc2)
    
    
    
    mds = MDS(2, max_iter=100, n_init=4, n_jobs=-1)
    btransformed22 = mds.fit_transform(fdataM)
    
    
    
    mds = MDS(2,max_iter=4000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=-1)

    #mds = MDS(2,max_iter=100, eps=1e-9, random_state=seed,
    #                   dissimilarity="precomputed", n_jobs=-1)
    
    temp=mds.fit(a)
    stress2=temp.stress_
    tdata = mds.fit_transform(a)

    model412=MDS(n_components=2, metric=False, max_iter=4000, eps=1e-12,
                        dissimilarity="precomputed", random_state=seed, n_jobs=-1,
                        n_init=1)
    pos2=temp.embedding_
    transformed22= model412.fit_transform(a, init=pos2)

    stress=temp.stress_
    # Rescale the data
    
    # Rotate the data
    #btransformed22=transformed22
    transformed22 *= np.sqrt((xx ** 2).sum()) / np.sqrt((transformed22 ** 2).sum())
    #transformed22=normalize(transformed22, axis=1, norm='l2')
    clf = PCA(n_components=2)
    transformed22 = clf.fit_transform(transformed22)
    # mycitiesFn=np.concatenate([mycities2, transformed22], axis=1)
    # color=fill_color(mycitiesFn)

    #print("stress value : ", stress7,stress2)
    etime=(time()-t0)/60
    print("Time min: " ,(time()-t0)/60)
    return tdata,pos2,btransformed22,transformed22,tdata7,etime,a
    #TEST

def doISO(fdata,fdataM,xx):
    import numpy as np
    t0=time()
    seed = np.random.seed(seed=3)
    #seed=1
    #CALCULATE DIF MATRICES

    #a=squareform(pdist(fdata, (lambda u,v: haversine(u,v))))
#     distance_matrixX  = squareform(pdist(xx,'sqeuclidean'))
#     distance_matrix2  = squareform(pdist(fdataM,'sqeuclidean'))
#     distance_matrix7  = squareform(pdist(gvdata,'sqeuclidean'))
#     distance_matrix = squareform(pdist(fdata2,'sqeuclidean'))
    #distance_matrixc = squareform(pdist(fdatac,'sqeuclidean'))
    #distance_matrixc2  = euclidean_distances(fdataM.astype(np.float64))
    a=xx
    distance_matrixc2=fdataM.astype(np.float64)

    mds7 = Isomap(2,max_iter=100, 
                        n_jobs=-1)
    temp7=mds7.fit(distance_matrixc2)
    #stress7=temp7.stress_
    tdata7 = mds7.fit_transform(distance_matrixc2)
    
    
    
    mds = Isomap(2, max_iter=100, n_jobs=-1)
    btransformed22 = mds.fit_transform(fdataM)
    
    
    
    mds = Isomap(2,max_iter=4000,  n_jobs=-1)

    #mds = MDS(2,max_iter=100, eps=1e-9, random_state=seed,
    #                   dissimilarity="precomputed", n_jobs=-1)
    
    temp=mds.fit(a)
    #stress2=temp.stress_
    tdata = mds.fit_transform(a)

    model412=Isomap(n_components=2,  max_iter=4000,   n_jobs=-1)
    pos2=temp.embedding_
    transformed22= model412.fit_transform(a)

    #stress=temp.stress_
    # Rescale the data
    
    # Rotate the data
    btransformed22=transformed22
    transformed22 *= np.sqrt((xx ** 2).sum()) / np.sqrt((transformed22 ** 2).sum())
    #transformed22=normalize(transformed22, axis=1, norm='l2')
    clf = PCA(n_components=2)
    transformed22 = clf.fit_transform(transformed22)
    # mycitiesFn=np.concatenate([mycities2, transformed22], axis=1)
    # color=fill_color(mycitiesFn)

    #print("stress value : ", stress7,stress2)
    print("Time min: " ,(time()-t0)/60)
    return tdata,pos2,btransformed22,transformed22,tdata7
    #TEST

def doTSNE(fdata,fdataM,xx):

    import numpy as np
    t0=time()
    seed = np.random.seed(seed=3)
    #seed=1
    #CALCULATE DIF MATRICES

    a=squareform(pdist(fdata, (lambda u,v: haversine(u,v))))
#     distance_matrixX  = squareform(pdist(xx,'sqeuclidean'))
#     distance_matrix2  = squareform(pdist(fdataM,'sqeuclidean'))
#     distance_matrix7  = squareform(pdist(gvdata,'sqeuclidean'))
#     distance_matrix = squareform(pdist(fdata2,'sqeuclidean'))
    #distance_matrixc = squareform(pdist(fdatac,'sqeuclidean'))
    distance_matrixc2  = euclidean_distances(fdataM.astype(np.float64))


    mds7 = TSNE(2, metric="precomputed", random_state=seed)
    temp7=mds7.fit(distance_matrixc2)

    tdata7 = mds7.fit_transform(distance_matrixc2)
    
    
    
    mds = TSNE(2, init='pca',random_state=seed)
    btransformed22 = mds.fit_transform(fdataM)
    
    
    
    mds = TSNE(2, random_state=seed,
                       metric="precomputed")

    #mds = MDS(2,max_iter=100, eps=1e-9, random_state=seed,
    #                   dissimilarity="precomputed", n_jobs=-1)
    
    temp=mds.fit(a)
    #stress2=temp.stress_
    tdata = mds.fit_transform(a)

    model412=TSNE(n_components=2, 
                        metric="precomputed", random_state=seed)
    pos2=temp.embedding_
    transformed22= model412.fit_transform(a)

    #stress=temp.stress_
    # Rescale the data
    
    # Rotate the data
    btransformed22=transformed22
    transformed22 *= np.sqrt((xx ** 2).sum()) / np.sqrt((transformed22 ** 2).sum())
    #transformed22=normalize(transformed22, axis=1, norm='l2')
    clf = PCA(n_components=2)
    transformed22 = clf.fit_transform(transformed22)
    # mycitiesFn=np.concatenate([mycities2, transformed22], axis=1)
    # color=fill_color(mycitiesFn)

    #print("stress value : ", stress7,stress2)
    print("Time min: " ,(time()-t0)/60)
    return tdata,pos2,btransformed22,transformed22,tdata7
    #TEST
def doLLE(fdata,fdataM,xx):
    from sklearn import manifold, datasets
    import numpy as np
    t0=time()
    seed = np.random.seed(seed=3)
    #seed=1
    #CALCULATE DIF MATRICES

    #a=squareform(pdist(fdata, (lambda u,v: haversine(u,v))))
#     distance_matrixX  = squareform(pdist(xx,'sqeuclidean'))
#     distance_matrix2  = squareform(pdist(fdataM,'sqeuclidean'))
#     distance_matrix7  = squareform(pdist(gvdata,'sqeuclidean'))
#     distance_matrix = squareform(pdist(fdata2,'sqeuclidean'))
    #distance_matrixc = squareform(pdist(fdatac,'sqeuclidean'))
    #distance_matrixc2  = euclidean_distances(fdataM.astype(np.float64))
    a=xx
    distance_matrixc2=fdataM.astype(np.float64)
    
  
    
    tdata7 =manifold.locally_linear_embedding(distance_matrixc2, n_neighbors=12,
                                             n_components=2,n_jobs=-1)
    
    
    
    btransformed22 =manifold.locally_linear_embedding(fdataM, n_neighbors=12,
                                             n_components=2,n_jobs=-1)
    
    tdata =manifold.locally_linear_embedding(a, n_neighbors=12,
                                             n_components=2,n_jobs=-1)
    
    mds = Isomap(2,max_iter=4000,  n_jobs=-1)

    #mds = MDS(2,max_iter=100, eps=1e-9, random_state=seed,
    #                   dissimilarity="precomputed", n_jobs=-1)
    
    temp=mds.fit(a)
    #stress2=temp.stress_
    tdata = mds.fit_transform(a)

    model412=Isomap(n_components=2,  max_iter=4000,   n_jobs=-1)
    pos2=temp.embedding_
    transformed22= model412.fit_transform(a)

    #stress=temp.stress_
    # Rescale the data
    
    # Rotate the data
    btransformed22=transformed22
    transformed22 *= np.sqrt((xx ** 2).sum()) / np.sqrt((transformed22 ** 2).sum())
    #transformed22=normalize(transformed22, axis=1, norm='l2')
    clf = PCA(n_components=2)
    transformed22 = clf.fit_transform(transformed22)
    # mycitiesFn=np.concatenate([mycities2, transformed22], axis=1)
    # color=fill_color(mycitiesFn)

    #print("stress value : ", stress7,stress2)
    print("Time min: " ,(time()-t0)/60)
    return tdata,pos2,btransformed22,transformed22,tdata
    #TEST

countries = [
{'timezones': ['Europe/Andorra'], 'code': 'AD', 'continent': 'Europe', 'name': 'Andorra', 'capital': 'Andorra la Vella'},
{'timezones': ['Asia/Kabul'], 'code': 'AF', 'continent': 'Asia', 'name': 'Afghanistan', 'capital': 'Kabul'},
{'timezones': ['America/Antigua'], 'code': 'AG', 'continent': 'North America', 'name': 'Antigua and Barbuda', 'capital': "St. John's"},
{'timezones': ['Europe/Tirane'], 'code': 'AL', 'continent': 'Europe', 'name': 'Albania', 'capital': 'Tirana'},
{'timezones': ['Asia/Yerevan'], 'code': 'AM', 'continent': 'Asia', 'name': 'Armenia', 'capital': 'Yerevan'},
{'timezones': ['Africa/Luanda'], 'code': 'AO', 'continent': 'Africa', 'name': 'Angola', 'capital': 'Luanda'},
{'timezones': ['America/Argentina/Buenos_Aires', 'America/Argentina/Cordoba', 'America/Argentina/Jujuy', 'America/Argentina/Tucuman', 'America/Argentina/Catamarca', 'America/Argentina/La_Rioja', 'America/Argentina/San_Juan', 'America/Argentina/Mendoza', 'America/Argentina/Rio_Gallegos', 'America/Argentina/Ushuaia'], 'code': 'AR', 'continent': 'South America', 'name': 'Argentina', 'capital': 'Buenos Aires'},
{'timezones': ['Europe/Vienna'], 'code': 'AT', 'continent': 'Europe', 'name': 'Austria', 'capital': 'Vienna'},
{'timezones': ['Australia/Lord_Howe', 'Australia/Hobart', 'Australia/Currie', 'Australia/Melbourne', 'Australia/Sydney', 'Australia/Broken_Hill', 'Australia/Brisbane', 'Australia/Lindeman', 'Australia/Adelaide', 'Australia/Darwin', 'Australia/Perth'], 'code': 'AU', 'continent': 'Oceania', 'name': 'Australia', 'capital': 'Canberra'},
{'timezones': ['Asia/Baku'], 'code': 'AZ', 'continent': 'Asia', 'name': 'Azerbaijan', 'capital': 'Baku'},
{'timezones': ['America/Barbados'], 'code': 'BB', 'continent': 'North America', 'name': 'Barbados', 'capital': 'Bridgetown'},
{'timezones': ['Asia/Dhaka'], 'code': 'BD', 'continent': 'Asia', 'name': 'Bangladesh', 'capital': 'Dhaka'},
{'timezones': ['Europe/Brussels'], 'code': 'BE', 'continent': 'Europe', 'name': 'Belgium', 'capital': 'Brussels'},
{'timezones': ['Africa/Ouagadougou'], 'code': 'BF', 'continent': 'Africa', 'name': 'Burkina Faso', 'capital': 'Ouagadougou'},
{'timezones': ['Europe/Sofia'], 'code': 'BG', 'continent': 'Europe', 'name': 'Bulgaria', 'capital': 'Sofia'},
{'timezones': ['Asia/Bahrain'], 'code': 'BH', 'continent': 'Asia', 'name': 'Bahrain', 'capital': 'Manama'},
{'timezones': ['Africa/Bujumbura'], 'code': 'BI', 'continent': 'Africa', 'name': 'Burundi', 'capital': 'Bujumbura'},
{'timezones': ['Africa/Porto-Novo'], 'code': 'BJ', 'continent': 'Africa', 'name': 'Benin', 'capital': 'Porto-Novo'},
{'timezones': ['Asia/Brunei'], 'code': 'BN', 'continent': 'Asia', 'name': 'Brunei Darussalam', 'capital': 'Bandar Seri Begawan'},
{'timezones': ['America/La_Paz'], 'code': 'BO', 'continent': 'South America', 'name': 'Bolivia', 'capital': 'Sucre'},
{'timezones': ['America/Noronha', 'America/Belem', 'America/Fortaleza', 'America/Recife', 'America/Araguaina', 'America/Maceio', 'America/Bahia', 'America/Sao_Paulo', 'America/Campo_Grande', 'America/Cuiaba', 'America/Porto_Velho', 'America/Boa_Vista', 'America/Manaus', 'America/Eirunepe', 'America/Rio_Branco'], 'code': 'BR', 'continent': 'South America', 'name': 'Brazil', 'capital': 'Bras\xc3\xadlia'},
{'timezones': ['America/Nassau'], 'code': 'BS', 'continent': 'North America', 'name': 'Bahamas', 'capital': 'Nassau'},
{'timezones': ['Asia/Thimphu'], 'code': 'BT', 'continent': 'Asia', 'name': 'Bhutan', 'capital': 'Thimphu'},
{'timezones': ['Africa/Gaborone'], 'code': 'BW', 'continent': 'Africa', 'name': 'Botswana', 'capital': 'Gaborone'},
{'timezones': ['Europe/Minsk'], 'code': 'BY', 'continent': 'Europe', 'name': 'Belarus', 'capital': 'Minsk'},
{'timezones': ['America/Belize'], 'code': 'BZ', 'continent': 'North America', 'name': 'Belize', 'capital': 'Belmopan'},
{'timezones': ['America/St_Johns', 'America/Halifax', 'America/Glace_Bay', 'America/Moncton', 'America/Goose_Bay', 'America/Blanc-Sablon', 'America/Montreal', 'America/Toronto', 'America/Nipigon', 'America/Thunder_Bay', 'America/Pangnirtung', 'America/Iqaluit', 'America/Atikokan', 'America/Rankin_Inlet', 'America/Winnipeg', 'America/Rainy_River', 'America/Cambridge_Bay', 'America/Regina', 'America/Swift_Current', 'America/Edmonton', 'America/Yellowknife', 'America/Inuvik', 'America/Dawson_Creek', 'America/Vancouver', 'America/Whitehorse', 'America/Dawson'], 'code': 'CA', 'continent': 'North America', 'name': 'Canada', 'capital': 'Ottawa'},
{'timezones': ['Africa/Kinshasa', 'Africa/Lubumbashi'], 'code': 'CD', 'continent': 'Africa', 'name': 'Democratic Republic of the Congo', 'capital': 'Kinshasa'},
{'timezones': ['Africa/Brazzaville'], 'code': 'CG', 'continent': 'Africa', 'name': 'Republic of the Congo', 'capital': 'Brazzaville'},
{'timezones': ['Africa/Abidjan'], 'code': 'CI', 'continent': 'Africa', 'name': "C\xc3\xb4te d'Ivoire", 'capital': 'Yamoussoukro'},
{'timezones': ['America/Santiago', 'Pacific/Easter'], 'code': 'CL', 'continent': 'South America', 'name': 'Chile', 'capital': 'Santiago'},
{'timezones': ['Africa/Douala'], 'code': 'CM', 'continent': 'Africa', 'name': 'Cameroon', 'capital': 'Yaound\xc3\xa9'},
{'timezones': ['Asia/Shanghai', 'Asia/Harbin', 'Asia/Chongqing', 'Asia/Urumqi', 'Asia/Kashgar'], 'code': 'CN', 'continent': 'Asia', 'name': "People's Republic of China", 'capital': 'Beijing'},
{'timezones': ['America/Bogota'], 'code': 'CO', 'continent': 'South America', 'name': 'Colombia', 'capital': 'Bogot\xc3\xa1'},
{'timezones': ['America/Costa_Rica'], 'code': 'CR', 'continent': 'North America', 'name': 'Costa Rica', 'capital': 'San Jos\xc3\xa9'},
{'timezones': ['America/Havana'], 'code': 'CU', 'continent': 'North America', 'name': 'Cuba', 'capital': 'Havana'},
{'timezones': ['Atlantic/Cape_Verde'], 'code': 'CV', 'continent': 'Africa', 'name': 'Cape Verde', 'capital': 'Praia'},
{'timezones': ['Asia/Nicosia'], 'code': 'CY', 'continent': 'Asia', 'name': 'Cyprus', 'capital': 'Nicosia'},
{'timezones': ['Europe/Prague'], 'code': 'CZ', 'continent': 'Europe', 'name': 'Czech Republic', 'capital': 'Prague'},
{'timezones': ['Europe/Berlin'], 'code': 'DE', 'continent': 'Europe', 'name': 'Germany', 'capital': 'Berlin'},
{'timezones': ['Africa/Djibouti'], 'code': 'DJ', 'continent': 'Africa', 'name': 'Djibouti', 'capital': 'Djibouti City'},
{'timezones': ['Europe/Copenhagen'], 'code': 'DK', 'continent': 'Europe', 'name': 'Denmark', 'capital': 'Copenhagen'},
{'timezones': ['America/Dominica'], 'code': 'DM', 'continent': 'North America', 'name': 'Dominica', 'capital': 'Roseau'},
{'timezones': ['America/Santo_Domingo'], 'code': 'DO', 'continent': 'North America', 'name': 'Dominican Republic', 'capital': 'Santo Domingo'},
{'timezones': ['America/Guayaquil', 'Pacific/Galapagos'], 'code': 'EC', 'continent': 'South America', 'name': 'Ecuador', 'capital': 'Quito'},
{'timezones': ['Europe/Tallinn'], 'code': 'EE', 'continent': 'Europe', 'name': 'Estonia', 'capital': 'Tallinn'},
{'timezones': ['Africa/Cairo'], 'code': 'EG', 'continent': 'Africa', 'name': 'Egypt', 'capital': 'Cairo'},
{'timezones': ['Africa/Asmera'], 'code': 'ER', 'continent': 'Africa', 'name': 'Eritrea', 'capital': 'Asmara'},
{'timezones': ['Africa/Addis_Ababa'], 'code': 'ET', 'continent': 'Africa', 'name': 'Ethiopia', 'capital': 'Addis Ababa'},
{'timezones': ['Europe/Helsinki'], 'code': 'FI', 'continent': 'Europe', 'name': 'Finland', 'capital': 'Helsinki'},
{'timezones': ['Pacific/Fiji'], 'code': 'FJ', 'continent': 'Oceania', 'name': 'Fiji', 'capital': 'Suva'},
{'timezones': ['Europe/Paris'], 'code': 'FR', 'continent': 'Europe', 'name': 'France', 'capital': 'Paris'},
{'timezones': ['Africa/Libreville'], 'code': 'GA', 'continent': 'Africa', 'name': 'Gabon', 'capital': 'Libreville'},
{'timezones': ['Asia/Tbilisi'], 'code': 'GE', 'continent': 'Asia', 'name': 'Georgia', 'capital': 'Tbilisi'},
{'timezones': ['Africa/Accra'], 'code': 'GH', 'continent': 'Africa', 'name': 'Ghana', 'capital': 'Accra'},
{'timezones': ['Africa/Banjul'], 'code': 'GM', 'continent': 'Africa', 'name': 'The Gambia', 'capital': 'Banjul'},
{'timezones': ['Africa/Conakry'], 'code': 'GN', 'continent': 'Africa', 'name': 'Guinea', 'capital': 'Conakry'},
{'timezones': ['Europe/Athens'], 'code': 'GR', 'continent': 'Europe', 'name': 'Greece', 'capital': 'Athens'},
{'timezones': ['America/Guatemala'], 'code': 'GT', 'continent': 'North America', 'name': 'Guatemala', 'capital': 'Guatemala City'},
{'timezones': ['America/Guatemala'], 'code': 'HT', 'continent': 'North America', 'name': 'Haiti', 'capital': 'Port-au-Prince'},
{'timezones': ['Africa/Bissau'], 'code': 'GW', 'continent': 'Africa', 'name': 'Guinea-Bissau', 'capital': 'Bissau'},
{'timezones': ['America/Guyana'], 'code': 'GY', 'continent': 'South America', 'name': 'Guyana', 'capital': 'Georgetown'},
{'timezones': ['America/Tegucigalpa'], 'code': 'HN', 'continent': 'North America', 'name': 'Honduras', 'capital': 'Tegucigalpa'},
{'timezones': ['Europe/Budapest'], 'code': 'HU', 'continent': 'Europe', 'name': 'Hungary', 'capital': 'Budapest'},
{'timezones': ['Asia/Jakarta', 'Asia/Pontianak', 'Asia/Makassar', 'Asia/Jayapura'], 'code': 'ID', 'continent': 'Asia', 'name': 'Indonesia', 'capital': 'Jakarta'},
{'timezones': ['Europe/Dublin'], 'code': 'IE', 'continent': 'Europe', 'name': 'Republic of Ireland', 'capital': 'Dublin'},
{'timezones': ['Asia/Jerusalem'], 'code': 'IL', 'continent': 'Asia', 'name': 'Israel', 'capital': 'Jerusalem'},
{'timezones': ['Asia/Calcutta'], 'code': 'IN', 'continent': 'Asia', 'name': 'India', 'capital': 'New Delhi'},
{'timezones': ['Asia/Baghdad'], 'code': 'IQ', 'continent': 'Asia', 'name': 'Iraq', 'capital': 'Baghdad'},
{'timezones': ['Asia/Tehran'], 'code': 'IR', 'continent': 'Asia', 'name': 'Iran', 'capital': 'Tehran'},
{'timezones': ['Atlantic/Reykjavik'], 'code': 'IS', 'continent': 'Europe', 'name': 'Iceland', 'capital': 'Reykjav\xc3\xadk'},
{'timezones': ['Europe/Rome'], 'code': 'IT', 'continent': 'Europe', 'name': 'Italy', 'capital': 'Rome'},
{'timezones': ['America/Jamaica'], 'code': 'JM', 'continent': 'North America', 'name': 'Jamaica', 'capital': 'Kingston'},
{'timezones': ['Asia/Amman'], 'code': 'JO', 'continent': 'Asia', 'name': 'Jordan', 'capital': 'Amman'},
{'timezones': ['Asia/Tokyo'], 'code': 'JP', 'continent': 'Asia', 'name': 'Japan', 'capital': 'Tokyo'},
{'timezones': ['Africa/Nairobi'], 'code': 'KE', 'continent': 'Africa', 'name': 'Kenya', 'capital': 'Nairobi'},
{'timezones': ['Asia/Bishkek'], 'code': 'KG', 'continent': 'Asia', 'name': 'Kyrgyzstan', 'capital': 'Bishkek'},
{'timezones': ['Pacific/Tarawa', 'Pacific/Enderbury', 'Pacific/Kiritimati'], 'code': 'KI', 'continent': 'Oceania', 'name': 'Kiribati', 'capital': 'Tarawa'},
{'timezones': ['Asia/Pyongyang'], 'code': 'KP', 'continent': 'Asia', 'name': 'North Korea', 'capital': 'Pyongyang'},
{'timezones': ['Asia/Seoul'], 'code': 'KR', 'continent': 'Asia', 'name': 'South Korea', 'capital': 'Seoul'},
{'timezones': ['Asia/Kuwait'], 'code': 'KW', 'continent': 'Asia', 'name': 'Kuwait', 'capital': 'Kuwait City'},
{'timezones': ['Asia/Beirut'], 'code': 'LB', 'continent': 'Asia', 'name': 'Lebanon', 'capital': 'Beirut'},
{'timezones': ['Europe/Vaduz'], 'code': 'LI', 'continent': 'Europe', 'name': 'Liechtenstein', 'capital': 'Vaduz'},
{'timezones': ['Africa/Monrovia'], 'code': 'LR', 'continent': 'Africa', 'name': 'Liberia', 'capital': 'Monrovia'},
{'timezones': ['Africa/Maseru'], 'code': 'LS', 'continent': 'Africa', 'name': 'Lesotho', 'capital': 'Maseru'},
{'timezones': ['Europe/Vilnius'], 'code': 'LT', 'continent': 'Europe', 'name': 'Lithuania', 'capital': 'Vilnius'},
{'timezones': ['Europe/Luxembourg'], 'code': 'LU', 'continent': 'Europe', 'name': 'Luxembourg', 'capital': 'Luxembourg City'},
{'timezones': ['Europe/Riga'], 'code': 'LV', 'continent': 'Europe', 'name': 'Latvia', 'capital': 'Riga'},
{'timezones': ['Africa/Tripoli'], 'code': 'LY', 'continent': 'Africa', 'name': 'Libya', 'capital': 'Tripoli'},
{'timezones': ['Indian/Antananarivo'], 'code': 'MG', 'continent': 'Africa', 'name': 'Madagascar', 'capital': 'Antananarivo'},
{'timezones': ['Pacific/Majuro', 'Pacific/Kwajalein'], 'code': 'MH', 'continent': 'Oceania', 'name': 'Marshall Islands', 'capital': 'Majuro'},
{'timezones': ['Europe/Skopje'], 'code': 'MK', 'continent': 'Europe', 'name': 'Macedonia', 'capital': 'Skopje'},
{'timezones': ['Africa/Bamako'], 'code': 'ML', 'continent': 'Africa', 'name': 'Mali', 'capital': 'Bamako'},
{'timezones': ['Asia/Rangoon'], 'code': 'MM', 'continent': 'Asia', 'name': 'Myanmar', 'capital': 'Naypyidaw'},
{'timezones': ['Asia/Ulaanbaatar', 'Asia/Hovd', 'Asia/Choibalsan'], 'code': 'MN', 'continent': 'Asia', 'name': 'Mongolia', 'capital': 'Ulaanbaatar'},
{'timezones': ['Africa/Nouakchott'], 'code': 'MR', 'continent': 'Africa', 'name': 'Mauritania', 'capital': 'Nouakchott'},
{'timezones': ['Europe/Malta'], 'code': 'MT', 'continent': 'Europe', 'name': 'Malta', 'capital': 'Valletta'},
{'timezones': ['Indian/Mauritius'], 'code': 'MU', 'continent': 'Africa', 'name': 'Mauritius', 'capital': 'Port Louis'},
{'timezones': ['Indian/Maldives'], 'code': 'MV', 'continent': 'Asia', 'name': 'Maldives', 'capital': 'Mal\xc3\xa9'},
{'timezones': ['Africa/Blantyre'], 'code': 'MW', 'continent': 'Africa', 'name': 'Malawi', 'capital': 'Lilongwe'},
{'timezones': ['America/Mexico_City', 'America/Cancun', 'America/Merida', 'America/Monterrey', 'America/Mazatlan', 'America/Chihuahua', 'America/Hermosillo', 'America/Tijuana'], 'code': 'MX', 'continent': 'North America', 'name': 'Mexico', 'capital': 'Mexico City'},
{'timezones': ['Asia/Kuala_Lumpur', 'Asia/Kuching'], 'code': 'MY', 'continent': 'Asia', 'name': 'Malaysia', 'capital': 'Kuala Lumpur'},
{'timezones': ['Africa/Maputo'], 'code': 'MZ', 'continent': 'Africa', 'name': 'Mozambique', 'capital': 'Maputo'},
{'timezones': ['Africa/Windhoek'], 'code': 'NA', 'continent': 'Africa', 'name': 'Namibia', 'capital': 'Windhoek'},
{'timezones': ['Africa/Niamey'], 'code': 'NE', 'continent': 'Africa', 'name': 'Niger', 'capital': 'Niamey'},
{'timezones': ['Africa/Lagos'], 'code': 'NG', 'continent': 'Africa', 'name': 'Nigeria', 'capital': 'Abuja'},
{'timezones': ['America/Managua'], 'code': 'NI', 'continent': 'North America', 'name': 'Nicaragua', 'capital': 'Managua'},
{'timezones': ['Europe/Amsterdam'], 'code': 'NL', 'continent': 'Europe', 'name': 'Kingdom of the Netherlands', 'capital': 'Amsterdam'},
{'timezones': ['Europe/Oslo'], 'code': 'NO', 'continent': 'Europe', 'name': 'Norway', 'capital': 'Oslo'},
{'timezones': ['Asia/Katmandu'], 'code': 'NP', 'continent': 'Asia', 'name': 'Nepal', 'capital': 'Kathmandu'},
{'timezones': ['Pacific/Nauru'], 'code': 'NR', 'continent': 'Oceania', 'name': 'Nauru', 'capital': 'Yaren'},
{'timezones': ['Pacific/Auckland', 'Pacific/Chatham'], 'code': 'NZ', 'continent': 'Oceania', 'name': 'New Zealand', 'capital': 'Wellington'},
{'timezones': ['Asia/Muscat'], 'code': 'OM', 'continent': 'Asia', 'name': 'Oman', 'capital': 'Muscat'},
{'timezones': ['America/Panama'], 'code': 'PA', 'continent': 'North America', 'name': 'Panama', 'capital': 'Panama City'},
{'timezones': ['America/Lima'], 'code': 'PE', 'continent': 'South America', 'name': 'Peru', 'capital': 'Lima'},
{'timezones': ['Pacific/Port_Moresby'], 'code': 'PG', 'continent': 'Oceania', 'name': 'Papua New Guinea', 'capital': 'Port Moresby'},
{'timezones': ['Asia/Manila'], 'code': 'PH', 'continent': 'Asia', 'name': 'Philippines', 'capital': 'Manila'},
{'timezones': ['Asia/Karachi'], 'code': 'PK', 'continent': 'Asia', 'name': 'Pakistan', 'capital': 'Islamabad'},
{'timezones': ['Europe/Warsaw'], 'code': 'PL', 'continent': 'Europe', 'name': 'Poland', 'capital': 'Warsaw'},
{'timezones': ['Europe/Lisbon', 'Atlantic/Madeira', 'Atlantic/Azores'], 'code': 'PT', 'continent': 'Europe', 'name': 'Portugal', 'capital': 'Lisbon'},
{'timezones': ['Pacific/Palau'], 'code': 'PW', 'continent': 'Oceania', 'name': 'Palau', 'capital': 'Ngerulmud'},
{'timezones': ['America/Asuncion'], 'code': 'PY', 'continent': 'South America', 'name': 'Paraguay', 'capital': 'Asunci\xc3\xb3n'},
{'timezones': ['Asia/Qatar'], 'code': 'QA', 'continent': 'Asia', 'name': 'Qatar', 'capital': 'Doha'},
{'timezones': ['Europe/Bucharest'], 'code': 'RO', 'continent': 'Europe', 'name': 'Romania', 'capital': 'Bucharest'},
{'timezones': ['Europe/Kaliningrad', 'Europe/Moscow', 'Europe/Volgograd', 'Europe/Samara', 'Asia/Yekaterinburg', 'Asia/Omsk', 'Asia/Novosibirsk', 'Asia/Krasnoyarsk', 'Asia/Irkutsk', 'Asia/Yakutsk', 'Asia/Vladivostok', 'Asia/Sakhalin', 'Asia/Magadan', 'Asia/Kamchatka', 'Asia/Anadyr'], 'code': 'RU', 'continent': 'Europe', 'name': 'Russia', 'capital': 'Moscow'},
{'timezones': ['Africa/Kigali'], 'code': 'RW', 'continent': 'Africa', 'name': 'Rwanda', 'capital': 'Kigali'},
{'timezones': ['Asia/Riyadh'], 'code': 'SA', 'continent': 'Asia', 'name': 'Saudi Arabia', 'capital': 'Riyadh'},
{'timezones': ['Pacific/Guadalcanal'], 'code': 'SB', 'continent': 'Oceania', 'name': 'Solomon Islands', 'capital': 'Honiara'},
{'timezones': ['Indian/Mahe'], 'code': 'SC', 'continent': 'Africa', 'name': 'Seychelles', 'capital': 'Victoria'},
{'timezones': ['Africa/Khartoum'], 'code': 'SD', 'continent': 'Africa', 'name': 'Sudan', 'capital': 'Khartoum'},
{'timezones': ['Europe/Stockholm'], 'code': 'SE', 'continent': 'Europe', 'name': 'Sweden', 'capital': 'Stockholm'},
{'timezones': ['Asia/Singapore'], 'code': 'SG', 'continent': 'Asia', 'name': 'Singapore', 'capital': 'Singapore'},
{'timezones': ['Europe/Ljubljana'], 'code': 'SI', 'continent': 'Europe', 'name': 'Slovenia', 'capital': 'Ljubljana'},
{'timezones': ['Europe/Bratislava'], 'code': 'SK', 'continent': 'Europe', 'name': 'Slovakia', 'capital': 'Bratislava'},
{'timezones': ['Africa/Freetown'], 'code': 'SL', 'continent': 'Africa', 'name': 'Sierra Leone', 'capital': 'Freetown'},
{'timezones': ['Europe/San_Marino'], 'code': 'SM', 'continent': 'Europe', 'name': 'San Marino', 'capital': 'San Marino'},
{'timezones': ['Africa/Dakar'], 'code': 'SN', 'continent': 'Africa', 'name': 'Senegal', 'capital': 'Dakar'},
{'timezones': ['Africa/Mogadishu'], 'code': 'SO', 'continent': 'Africa', 'name': 'Somalia', 'capital': 'Mogadishu'},
{'timezones': ['America/Paramaribo'], 'code': 'SR', 'continent': 'South America', 'name': 'Suriname', 'capital': 'Paramaribo'},
{'timezones': ['Africa/Sao_Tome'], 'code': 'ST', 'continent': 'Africa', 'name': 'S\xc3\xa3o Tom\xc3\xa9 and Pr\xc3\xadncipe', 'capital': 'S\xc3\xa3o Tom\xc3\xa9'},
{'timezones': ['Asia/Damascus'], 'code': 'SY', 'continent': 'Asia', 'name': 'Syria', 'capital': 'Damascus'},
{'timezones': ['Africa/Lome'], 'code': 'TG', 'continent': 'Africa', 'name': 'Togo', 'capital': 'Lom\xc3\xa9'},
{'timezones': ['Asia/Bangkok'], 'code': 'TH', 'continent': 'Asia', 'name': 'Thailand', 'capital': 'Bangkok'},
{'timezones': ['Asia/Dushanbe'], 'code': 'TJ', 'continent': 'Asia', 'name': 'Tajikistan', 'capital': 'Dushanbe'},
{'timezones': ['Asia/Ashgabat'], 'code': 'TM', 'continent': 'Asia', 'name': 'Turkmenistan', 'capital': 'Ashgabat'},
{'timezones': ['Africa/Tunis'], 'code': 'TN', 'continent': 'Africa', 'name': 'Tunisia', 'capital': 'Tunis'},
{'timezones': ['Pacific/Tongatapu'], 'code': 'TO', 'continent': 'Oceania', 'name': 'Tonga', 'capital': 'Nuku\xca\xbbalofa'},
{'timezones': ['Europe/Istanbul'], 'code': 'TR', 'continent': 'Asia', 'name': 'Turkey', 'capital': 'Ankara'},
{'timezones': ['America/Port_of_Spain'], 'code': 'TT', 'continent': 'North America', 'name': 'Trinidad and Tobago', 'capital': 'Port of Spain'},
{'timezones': ['Pacific/Funafuti'], 'code': 'TV', 'continent': 'Oceania', 'name': 'Tuvalu', 'capital': 'Funafuti'},
{'timezones': ['Africa/Dar_es_Salaam'], 'code': 'TZ', 'continent': 'Africa', 'name': 'Tanzania', 'capital': 'Dodoma'},
{'timezones': ['Europe/Kiev', 'Europe/Uzhgorod', 'Europe/Zaporozhye', 'Europe/Simferopol'], 'code': 'UA', 'continent': 'Europe', 'name': 'Ukraine', 'capital': 'Kiev'},
{'timezones': ['Africa/Kampala'], 'code': 'UG', 'continent': 'Africa', 'name': 'Uganda', 'capital': 'Kampala'},
{'timezones': ['America/New_York', 'America/Detroit', 'America/Kentucky/Louisville', 'America/Kentucky/Monticello', 'America/Indiana/Indianapolis', 'America/Indiana/Marengo', 'America/Indiana/Knox', 'America/Indiana/Vevay', 'America/Chicago', 'America/Indiana/Vincennes', 'America/Indiana/Petersburg', 'America/Menominee', 'America/North_Dakota/Center', 'America/North_Dakota/New_Salem', 'America/Denver', 'America/Boise', 'America/Shiprock', 'America/Phoenix', 'America/Los_Angeles', 'America/Anchorage', 'America/Juneau', 'America/Yakutat', 'America/Nome', 'America/Adak', 'Pacific/Honolulu'], 'code': 'US', 'continent': 'North America', 'name': 'United States', 'capital': 'Washington, D.C.'},
{'timezones': ['America/Montevideo'], 'code': 'UY', 'continent': 'South America', 'name': 'Uruguay', 'capital': 'Montevideo'},
{'timezones': ['Asia/Samarkand', 'Asia/Tashkent'], 'code': 'UZ', 'continent': 'Asia', 'name': 'Uzbekistan', 'capital': 'Tashkent'},
{'timezones': ['Europe/Vatican'], 'code': 'VA', 'continent': 'Europe', 'name': 'Vatican City', 'capital': 'Vatican City'},
{'timezones': ['America/Caracas'], 'code': 'VE', 'continent': 'South America', 'name': 'Venezuela', 'capital': 'Caracas'},
{'timezones': ['Asia/Saigon'], 'code': 'VN', 'continent': 'Asia', 'name': 'Vietnam', 'capital': 'Hanoi'},
{'timezones': ['Pacific/Efate'], 'code': 'VU', 'continent': 'Oceania', 'name': 'Vanuatu', 'capital': 'Port Vila'},
{'timezones': ['Asia/Aden'], 'code': 'YE', 'continent': 'Asia', 'name': 'Yemen', 'capital': "Sana'a"},
{'timezones': ['Africa/Lusaka'], 'code': 'ZM', 'continent': 'Africa', 'name': 'Zambia', 'capital': 'Lusaka'},
{'timezones': ['Africa/Harare'], 'code': 'ZW', 'continent': 'Africa', 'name': 'Zimbabwe', 'capital': 'Harare'},
{'timezones': ['Africa/Algiers'], 'code': 'DZ', 'continent': 'Africa', 'name': 'Algeria', 'capital': 'Algiers'},
{'timezones': ['Europe/Sarajevo'], 'code': 'BA', 'continent': 'Europe', 'name': 'Bosnia and Herzegovina', 'capital': 'Sarajevo'},
{'timezones': ['Asia/Phnom_Penh'], 'code': 'KH', 'continent': 'Asia', 'name': 'Cambodia', 'capital': 'Phnom Penh'},
{'timezones': ['Africa/Bangui'], 'code': 'CF', 'continent': 'Africa', 'name': 'Central African Republic', 'capital': 'Bangui'},
{'timezones': ['Africa/Ndjamena'], 'code': 'TD', 'continent': 'Africa', 'name': 'Chad', 'capital': "N'Djamena"},
{'timezones': ['Indian/Comoro'], 'code': 'KM', 'continent': 'Africa', 'name': 'Comoros', 'capital': 'Moroni'},
{'timezones': ['Europe/Zagreb'], 'code': 'HR', 'continent': 'Europe', 'name': 'Croatia', 'capital': 'Zagreb'},
{'timezones': ['Asia/Dili'], 'code': 'TL', 'continent': 'Asia', 'name': 'East Timor', 'capital': 'Dili'},
{'timezones': ['America/El_Salvador'], 'code': 'SV', 'continent': 'North America', 'name': 'El Salvador', 'capital': 'San Salvador'},
{'timezones': ['Africa/Malabo'], 'code': 'GQ', 'continent': 'Africa', 'name': 'Equatorial Guinea', 'capital': 'Malabo'},
{'timezones': ['America/Grenada'], 'code': 'GD', 'continent': 'North America', 'name': 'Grenada', 'capital': "St. George's"},
{'timezones': ['Asia/Almaty', 'Asia/Qyzylorda', 'Asia/Aqtobe', 'Asia/Aqtau', 'Asia/Oral'], 'code': 'KZ', 'continent': 'Asia', 'name': 'Kazakhstan', 'capital': 'Astana'},
{'timezones': ['Asia/Vientiane'], 'code': 'LA', 'continent': 'Asia', 'name': 'Laos', 'capital': 'Vientiane'},
{'timezones': ['Pacific/Truk', 'Pacific/Ponape', 'Pacific/Kosrae'], 'code': 'FM', 'continent': 'Oceania', 'name': 'Federated States of Micronesia', 'capital': 'Palikir'},
{'timezones': ['Europe/Chisinau'], 'code': 'MD', 'continent': 'Europe', 'name': 'Moldova', 'capital': 'Chi\xc5\x9fin\xc4\x83u'},
{'timezones': ['Europe/Monaco'], 'code': 'MC', 'continent': 'Europe', 'name': 'Monaco', 'capital': 'Monaco'},
{'timezones': ['Europe/Podgorica'], 'code': 'ME', 'continent': 'Europe', 'name': 'Montenegro', 'capital': 'Podgorica'},
{'timezones': ['Africa/Casablanca'], 'code': 'MA', 'continent': 'Africa', 'name': 'Morocco', 'capital': 'Rabat'},
{'timezones': ['America/St_Kitts'], 'code': 'KN', 'continent': 'North America', 'name': 'Saint Kitts and Nevis', 'capital': 'Basseterre'},
{'timezones': ['America/St_Lucia'], 'code': 'LC', 'continent': 'North America', 'name': 'Saint Lucia', 'capital': 'Castries'},
{'timezones': ['America/St_Vincent'], 'code': 'VC', 'continent': 'North America', 'name': 'Saint Vincent and the Grenadines', 'capital': 'Kingstown'},
{'timezones': ['Pacific/Apia'], 'code': 'WS', 'continent': 'Oceania', 'name': 'Samoa', 'capital': 'Apia'},
{'timezones': ['Europe/Belgrade'], 'code': 'RS', 'continent': 'Europe', 'name': 'Serbia', 'capital': 'Belgrade'},
{'timezones': ['Africa/Johannesburg'], 'code': 'ZA', 'continent': 'Africa', 'name': 'South Africa', 'capital': 'Pretoria'},
{'timezones': ['Europe/Madrid', 'Africa/Ceuta', 'Atlantic/Canary'], 'code': 'ES', 'continent': 'Europe', 'name': 'Spain', 'capital': 'Madrid'},
{'timezones': ['Asia/Colombo'], 'code': 'LK', 'continent': 'Asia', 'name': 'Sri Lanka', 'capital': 'Sri Jayewardenepura Kotte'},
{'timezones': ['Africa/Mbabane'], 'code': 'SZ', 'continent': 'Africa', 'name': 'Swaziland', 'capital': 'Mbabane'},
{'timezones': ['Europe/Zurich'], 'code': 'CH', 'continent': 'Europe', 'name': 'Switzerland', 'capital': 'Bern'},
{'timezones': ['Asia/Dubai'], 'code': 'AE', 'continent': 'Asia', 'name': 'United Arab Emirates', 'capital': 'Abu Dhabi'},
{'timezones': ['Europe/London'], 'code': 'GB', 'continent': 'Europe', 'name': 'United Kingdom', 'capital': 'London'},
{'timezones': ['Asia/Taipei'], 'code': 'TW', 'continent': 'Asia', 'name': 'Taiwan', 'capital': 'Taipei'},
{'timezones': ['Asia/Hong_Kong'], 'code': 'HK', 'continent': 'Asia', 'name': 'Hong Kong', 'capital': 'Hong Kong'},
{'timezones': ['CET'], 'code': 'XK', 'continent': 'Europe', 'name': 'Kosovo', 'capital': 'Pristina'},
{'timezones': ['America/Puerto_Rico'], 'code': 'PR', 'continent': 'America', 'name': 'Puerto Rico', 'capital': 'San Juan'},
]

