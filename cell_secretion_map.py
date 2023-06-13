import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import re
import matplotlib
import matplotlib.pyplot as plt
#import generateheatmap as Heatmap
#import generateNetwork as Network
import plotly.tools
import random
import plotly.express as px  
import json

from streamlit_cropper import st_cropper
#from PIL import Image
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.plotting import figure
from bokeh.palettes import Category10_4, Category20, Category20b, Category20c
from streamlit_bokeh_events import streamlit_bokeh_events
from bokeh.transform import factor_cmap
import plotly.express as px
st.set_page_config(layout='wide')
from matplotlib import cm
import base64
from scipy import stats
import requests
from st_aggrid import GridUpdateMode, DataReturnMode
from plotly.graph_objs import Figure
#from sklearn.preprocessing import LabelEncoder
from plotly.callbacks import Points, InputDeviceState
global value_counts

### color registeration ###
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as col

# 3D Heatmap in Python using matplotlib
# to make plot interactive
#%matplotlib inline

### color scale ###
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
def REGISTER_TERRAIN_MAP(**kwargs):
    full=np.power(10,4)*2
    alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 0
    t = int(full*alpha)
    top = cm.get_cmap('Blues_r', full)
    bottom = cm.get_cmap('Reds', full)
    fade = kwargs['fade'] if 'fade' in kwargs.keys() else [0,0,0,1]
    if alpha == 0:
        newcolors = np.vstack((
                top(np.linspace(0, 1, full))[0:int(full-t)],
               ([fade]),
               bottom(np.linspace(0, 1, full))[int(t):full]
            ))
    else:
        gray = cm.get_cmap('Greys', t)
        newcolors = np.vstack((
                        top(np.linspace(0, 1, full))[0:int(full-t)],
                           np.array([[0,0,0,0.5]]*(t-1)),
                        ([fade]),
                       np.array([[0,0,0,0.5]]*(t-1)),
                   bottom(np.linspace(0, 1, full))[int(t):full]
                    ))
        np.array([[1,2,3],]*3)
    newcmp = ListedColormap(newcolors, name='RedBlue')   
    return(newcmp)

# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
def plot_3D(mtx, **kwargs):
    fig = plt.figure(figsize=(10, 10))
    #ax = Axes3D(fig)
    ax = fig.add_subplot(121, projection='3d')
    mtx_pre = kwargs['previous'] if 'previous' in kwargs.keys() else pd.DataFrame(np.zeros(shape=(50, 50)))
    
    lx,ly= size(mtx,0),size(mtx,1)       # Work out matrix dimensions
    xpos = np.arange(0,lx,1)    # Set up a mesh of positions
    ypos = np.arange(0,ly,1)
    xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)
    
    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx*ly)
    
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    delta_dz = mtx.values.flatten() - mtx_pre.values.flatten()
    # delta negative to zero
    delta_dz[delta_dz<0] = 0
    #print(delta_dz)
    dz_pre = mtx_pre.values.flatten()
    #cs = ['r', 'g', 'b', 'y', 'c'] * ly   
    print(xpos.shape)
    print(ypos.shape)
    print(dz_pre.shape)
    _zpos = zpos   # the starting zpos for each bar
    ax.bar3d(xpos, ypos,_zpos, dx, dy, dz_pre, color='b', shade=True)
    _zpos += dz_pre
    ax.bar3d(xpos, ypos,_zpos, dx, dy, delta_dz, color='r', shade=True)
    ax.set_zlim(0,5000)
    azim = 120
    elev = 70
    dist = 10
    if azim is not None:
        ax.azim = azim
    if dist is not None:
        ax.dist = dist
    if elev is not None:
        ax.elev = elev
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.invert_xaxis()
    #sh()
    #ax.xaxis.set_ticklabels(mtx.columns) #w_xaxis.set_ticklabels
    #ax.yaxis.set_ticklabels(mtx.index)
    #ax.set_xlabel('Letter')
    #ax.set_ylabel('Day')
    #ax.set_zlabel('Occurrence')
        
    #plt.savefig('./myplot.png')
    #plt.close()
    
    ################################
    ### 3D terrain delta changes ###
    ################################
    ax = fig.add_subplot(122, projection='3d')
    #X = np.linspace(0 , lx , lx )
    #Y = np.linspace(0 , ly , ly )
    X, Y = np.arange(0,lx,1),np.arange(0,ly,1)    # Set up a mesh of positions
    X, Y = np.meshgrid(Y,X)   
    
    #print(lx)
    #print(ly)
    Z = np.array(mtx.reset_index(drop=True).subtract(mtx_pre.reset_index(drop=True)))
    Z[Z<0] = 0
    #print(Z.shape)
    #print(X.shape)
    #print(Y.shape)
    # Plot the surface.
    surf = ax.plot_surface(X,Y,Z, cmap=REGISTER_TERRAIN_MAP(),#vmax=Bound, vmin=-Bound,, antialiased=False
                           linewidth=0)  
    # Customize the z axis.
    #ax.set_zlim(-Bound, Bound)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    if azim is not None:
        ax.azim = azim
    if dist is not None:
        ax.dist = dist
    if elev is not None:
        ax.elev = elev
    # Hide grid lines
    ax.grid(False)
    ax.set_zlim(0,5000)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    
    #ax.xaxis.set_ticklabels(mtx.columns)
    #ax.yaxis.set_ticklabels(mtx.index)
    ax.invert_xaxis()
    plt.show(1)
    
    return(X,Y,Z)

def plot_inequality(mtx,**kwargs):
    x_unit = kwargs['x_unit'] if 'x_unit' in kwargs.keys() else 50 
    y_unit = kwargs['y_unit'] if 'y_unit' in kwargs.keys() else 50 
    
    fig = plt.figure(figsize=(10, 5))
    ax1,ax2 = fig.add_subplot(121),fig.add_subplot(122)
    (x_val,y_val)=(mtx.sum(0),mtx.sum(1))
    ax1.plot(np.arange(0,1,1/x_unit),(x_val/x_val.sum()).cumsum()-np.arange(0,1,1/x_unit))
    ax1.plot(np.arange(0,1,1/x_unit),np.arange(0,1,1/x_unit)-np.arange(0,1,1/x_unit),color='Black')
    ax2.plot(np.arange(0,1,1/y_unit),(y_val/y_val.sum()).cumsum()-np.arange(0,1,1/y_unit))
    ax2.plot(np.arange(0,1,1/y_unit),np.arange(0,1,1/y_unit)-np.arange(0,1,1/y_unit),color='Black')
    plt.show()
    
    x_inequ = ((x_val/x_val.sum()).cumsum()-np.arange(0,1,1/x_unit)).sum()
    y_inequ = ((y_val/y_val.sum()).cumsum()-np.arange(0,1,1/y_unit)).sum()
    
    return(x_inequ,y_inequ,x_val,y_val)

########################
### set up new color ###
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
### custome the levels of the colors ###
customRange = list(np.linspace(500,5000,7))
full = len(customRange)
top = cm.get_cmap('rainbow', full)
newcolors = np.vstack(top(np.linspace(0, 1, full)),)
newcmp = ListedColormap(newcolors, name='newcmp')
########################

###############################################
### Create a LinearSegmentedColormap object ###
### Get masks using top-/bottom-rank values ###
def GET_MASK(z=[],top=50,btm=50):
    z_t10, z_b10 = np.sort(z)[-top], np.sort(z)[btm]
    mask, mask_top, mask_btm = (z>=z_t10) + (z<=z_b10), z>=z_t10, z<=z_b10
    return(mask,mask_top,mask_btm)

def DERIVATIVE(arr, dx):
    diff = np.diff(arr)    
    # Divide by dx to approximate the derivative
    derivative = diff / dx
    return derivative

### plot inequality ###
def GINI_IDX(mtx=[],x_val=[],y_val=[],x_unit=[],y_unit=[]):
    ### lorenz curve ###
    x_val,y_val=mtx.sum(0),mtx.sum(1)
    x_inequ = ((x_val/x_val.sum()).cumsum()-np.arange(0,1,1/x_unit))
    y_inequ = ((y_val/y_val.sum()).cumsum()-np.arange(0,1,1/y_unit))
    
    ### derivative of gini curve ###
    x_drvtv = list(DERIVATIVE(x_inequ,1))
    y_drvtv = list(DERIVATIVE(y_inequ,1))
    
    ### first point set to be zero ###
    x_drvtv.insert(0,0)
    y_drvtv.insert(0,0)
    
    return(x_drvtv,y_drvtv)

### major function of plot stream ###
# https://stackoverflow.com/questions/16529892/adding-water-flow-arrows-to-matplotlib-contour-plot
from scipy.interpolate import Rbf
def plotStream(mtx=[],lx=50,top=10,btm=10,**kwargs):
    lx,ly= size(mtx,0),size(mtx,1) # Work out matrix dimensions  
    cmlt_contour_color = kwargs['cmlt_contour_color'] if 'cmlt_contour_color' in kwargs.keys() else 'red'
    mtx_pre = kwargs['previous'] if 'previous' in kwargs.keys() else pd.DataFrame(np.zeros(shape=(50, 50)))
    x_unit = kwargs['x_unit'] if 'x_unit' in kwargs.keys() else 50 
    y_unit = kwargs['y_unit'] if 'y_unit' in kwargs.keys() else 50
    
    # -- Plot --------------------------
    fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(111)
    ax = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
    ax1 = plt.subplot2grid((4, 4), (3, 0), colspan=3, rowspan=1)
    ax2 = plt.subplot2grid((4, 4), (0, 3),colspan=1, rowspan=3)  
    
    ### delta changes ###
    Z = np.array(mtx.reset_index(drop=True).subtract(mtx_pre.reset_index(drop=True)))[:,::-1]     
    Z_cmltv = np.array(mtx.reset_index(drop=True))[:,::-1]
    Z[Z<0] = 0
    Z_cmltv[Z_cmltv<0] = 0
    
    ### Interpolate these onto a regular grid ###
    xi,yi = np.meshgrid(np.arange(0,lx,1),np.arange(0,ly,1)) # Set up a mesh of positions
    
    ### flatten the observation matrix ###
    xpos,ypos,z,z_cmltv = xi.flatten(),yi.flatten(),Z.flatten(),Z_cmltv.flatten() # Convert positions to 1D array

    ### get_mask ###
    (mask,mask_top,mask_btm) = GET_MASK(z=z,top=top,btm=btm)
    (mask_cmltv,mask_top_cmltv,mask_btm_cmltv) = GET_MASK(z=z_cmltv,top=top,btm=btm)

    func = Rbf(xpos[mask],ypos[mask], z[mask], function='linear')
    func_cmltv = Rbf(xpos[mask_cmltv],ypos[mask_cmltv], z_cmltv[mask_cmltv], function='linear')
    zi_cmltv,zi = func_cmltv(xi, yi),func(xi, yi) 
    
    # get the positions of the masked culmulative expression
    shape = (lx,ly)
    mask_top_cmltv_pd = pd.DataFrame(mask_top_cmltv.reshape(shape))
    mask_top_cmltv_pd_pstn = np.where(mask_top_cmltv_pd == True)
    
    ### Plot flowlines cmltv arrow plot ###
    if top != 0:
        dy, dx = np.gradient(-zi_cmltv) # Flow goes down gradient (thus -zi)
        stream = ax.streamplot(xi[0,:], yi[:,0], dx, dy, density=1,arrowsize=1,linewidth=3,minlength=0.011,
                           color = 'grey',broken_streamlines=True,
                           start_points=np.column_stack((mask_top_cmltv_pd_pstn[1],mask_top_cmltv_pd_pstn[0]))
                          )#color='0.6',
        # Customize the transparency level
        stream.lines.set_alpha(0.5)

    ### Contour gridded head observations
    contours = ax.contour(xi, yi, zi, 
                          linewidths=2,levels=list(np.linspace(500,5000,7)),cmap=newcmp,#list(np.linspace(500,5000,7))
                          linestyles='dashed')
    ax.clabel(contours)
    contours_cmltv = ax.contour(xi, yi, zi_cmltv, [1200], colors = cmlt_contour_color,alpha=0.5,linewidths=4) #
    ax.clabel(contours_cmltv,inline=True, fontsize=12)
    
    ### Plot well locations ###
    if top != 0:
        ax.plot(xpos[mask_top],ypos[mask_top], 'ko',color=cmlt_contour_color)
        #ax.plot(xpos[mask_top_cmltv],ypos[mask_top_cmltv], 'ko',color='black')
        #ax.plot(xpos[mask_btm],ypos[mask_btm], 'ko',color='blue')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.axis('off')
    
    (x_drvtv,y_drvtv) = GINI_IDX(mtx=mtx,x_val=x_val,y_val=y_val,x_unit=x_unit,y_unit=y_unit)
    ax1.plot(np.arange(0,1,1/x_unit),x_drvtv)
    ax1.plot(np.arange(0,1,1/x_unit),np.arange(0,1,1/x_unit)-np.arange(0,1,1/x_unit),color='Black')   
    
    ax2.plot(y_drvtv,np.arange(0,1,1/y_unit))
    ax2.plot(np.arange(0,1,1/y_unit)-np.arange(0,1,1/y_unit),np.arange(0,1,1/y_unit),color='Black')
    ax2.invert_yaxis()
    
    plt.show()
    return(x_inequ.sum(),y_inequ.sum(),x_drvtv,y_drvtv,fig)  

####################################################
### load the object and generate the coordincate ###
####################################################
@st.cache(allow_output_mutation=True)
def LOAD_DATA(inputDir='',dataDir=''):
    df = pd.read_excel(r'./'+inputDir+dataDir, sheet_name='Sheet12',header=None)
    return df

inputDir='input/'
dataDir="IL-6 with amount summary.xlsx"
df = LOAD_DATA(inputDir=inputDir,dataDir=dataDir)

#####################
### web interface ###
#####################

st.title('Cell Secretion Map')
st.markdown('**')
st.header("Comparison of anisotropy and isotropy")

### IL6 sheet 21
(top,btm) = ([0,2,2,2,2,2,3],[1,1,1,1,200,200,200])
diameter = 15
(x_start,x_end,y_start,y_end) = (25-diameter,25+diameter,25-diameter,25+diameter)

#mtx_pre = pd.DataFrame(np.zeros(shape=(x_end-x_start, y_end-y_start)))
#mtx_pre.columns = range(y_start,y_end)
#mtx_pre.index=range(x_start,x_end)
(x_unit,y_unit) = (x_end-x_start,y_end-y_start)
i=2
mtx = df.iloc[i*50:(i+1)*50,0:50]
mtx = mtx.iloc[x_start:x_end,y_start:y_end]
mtx_pre = df.iloc[(i-2)*50:(i-1)*50,0:50]      
mtx_pre = mtx_pre.iloc[x_start:x_end,y_start:y_end]
(x_inequ, y_inequ,x_drvtv,y_drvtv,fig) = plotStream(mtx=mtx,top=6,btm=200,previous = mtx_pre,
                                                x_unit = x_unit,y_unit = y_unit, 
                                                cmlt_contour_color = cmlt_contour_color[i])
st.plotly_chart(fig,use_container_width=True)  
st.write(pd.head(10))
