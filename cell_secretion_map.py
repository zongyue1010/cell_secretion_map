
### error debug ###


# pip install -U click==8


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
#from bokeh.models import ColumnDataSource, CustomJS
#from bokeh.plotting import figure
#from bokeh.palettes import Category10_4, Category20, Category20b, Category20c
#from streamlit_bokeh_events import streamlit_bokeh_events
#from bokeh.transform import factor_cmap
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

#@st.cache(allow_output_mutation=True)
def get_table_download_link(df, **kwargs):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True, sep ='\t')
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    prefix = "Download txt file for "
    if("fileName" in kwargs.keys()):
        prefix += kwargs['fileName']
    href = f'<a href="data:file/csv;base64,{b64}" download="'+kwargs['fileName']+'\.txt">'+prefix+'</a>'
    return(href)


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
    ax2 = fig.add_subplot(122, projection='3d')
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
    surf = ax2.plot_surface(X,Y,Z, cmap=REGISTER_TERRAIN_MAP(),#vmax=Bound, vmin=-Bound,, antialiased=False
                           linewidth=0)  
    # Customize the z axis.
    #ax.set_zlim(-Bound, Bound)
    ax2.zaxis.set_major_locator(LinearLocator(10))
    ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    if azim is not None:
        ax2.azim = azim
    if dist is not None:
        ax2.dist = dist
    if elev is not None:
        ax2.elev = elev
    # Hide grid lines
    ax2.grid(False)
    ax2.set_zlim(0,5000)
    ax2.set_xlabel('x-axis')
    ax2.set_ylabel('y-axis')
    
    #ax.xaxis.set_ticklabels(mtx.columns)
    #ax.yaxis.set_ticklabels(mtx.index)
    ax2.invert_xaxis()
    plt.show()
    st.pyplot(fig)
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
    st.pyplot()
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
                           color = 'grey',#broken_streamlines=True,
                           start_points=np.column_stack((mask_top_cmltv_pd_pstn[1],mask_top_cmltv_pd_pstn[0]))
                          )#color='0.6',
        # Customize the transparency level
        stream.lines.set_alpha(0.5)

    ### Contour gridded head observations
    try:
        contours = ax.contour(xi, yi, zi, 
                          linewidths=2,levels=list(np.linspace(500,5000,7)),cmap=newcmp,#list(np.linspace(500,5000,7))
                          linestyles='dashed')
        ax.clabel(contours)
    except:
        print("not available")
    try:
        contours_cmltv = ax.contour(xi, yi, zi_cmltv, [1200], colors = cmlt_contour_color,alpha=0.5,linewidths=4) #
        ax.clabel(contours_cmltv,inline=True, fontsize=12)
    except:
        print("not available")    
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
    
    (x_drvtv,y_drvtv) = GINI_IDX(mtx=mtx,x_unit=x_unit,y_unit=y_unit)
    ax1.plot(np.arange(0,1,1/x_unit),x_drvtv)
    ax1.plot(np.arange(0,1,1/x_unit),np.arange(0,1,1/x_unit)-np.arange(0,1,1/x_unit),color='Black')   
    
    ax2.plot(y_drvtv,np.arange(0,1,1/y_unit))
    ax2.plot(np.arange(0,1,1/y_unit)-np.arange(0,1,1/y_unit),np.arange(0,1,1/y_unit),color='Black')
    ax2.invert_yaxis()
  
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)
    return(x_drvtv,y_drvtv)   




####################################################
### load the object and generate the coordincate ###
####################################################

def LOAD_DATA(inputDir='',dataDir='',sheet_name=''):
    df = pd.read_excel(r'./'+inputDir+dataDir, sheet_name=sheet_name,header=None)
    return df

###############################
### sidemenu of data source ###
###############################
st.sidebar.subheader('Data source')
inputDir='input/'
from os import walk
f = []
for (dirpath, dirnames, filenames) in walk(r'./'+inputDir):
    f.extend(filenames)
    break
    
dataDir = st.sidebar.selectbox(
    'select a cell source:',
    tuple(f),key='workingdir'
    )
st.sidebar.markdown('You selected `%s`' % dataDir)

import openpyxl

# Load the Excel file
workbook = openpyxl.load_workbook(r'./'+inputDir+dataDir)

# Get a list of all sheet names
sheet_names = workbook.sheetnames


#####################
### web interface ###
#####################
from PIL import Image
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

st.image(add_logo(logo_path="./aipharm_logo.png", width=400, height=100)) 

st.title('An online interactive analytical platform for cell secretion map generation')
st.markdown('*Zongliang Yue, Lang Zhou, and Pengyu Chen*')
st.header("Comparison of anisotropy and isotropy cell secretion signals")

step1 = st.checkbox('Step 1: show the heatmap and delta changes of the cell signals',value=True)
cmlt_contour_color = ['blue','blue','blue','orange','orange','red','red']

drvt1_index_x_sets,drvt1_index_y_sets = [],[]
drvt2_index_x_sets,drvt2_index_y_sets = [],[]


   
####################
### main content ###
####################
col1, col2 = st.columns(2)
if step1:
    with st.form("form"):
        with col1:
        
            drvt1_index_x_sets = []
            drvt1_index_y_sets = []
            ### IL6 sheet 21 anisotropy
            # List of options for the select box
            #sheet_names1 = ['Sheet21-15max', 'Sheet20','Sheet19','Sheet18','Sheet17-15max','Sheet16','Sheet11','Sheet10-180c','Sheet9-180c',
             #              'Sheet8-15max','Sheet7-15 max','Sheet6-depends','Sheet5-depends','Sheet3','Sheet2','Sheet1']
            sheet_names1 = sheet_names 
            selected_option1 = st.selectbox('Select an option', sheet_names1,key='sheet_names1')
            df1 = LOAD_DATA(inputDir=inputDir,dataDir=dataDir,sheet_name=selected_option1)
            
            hs1_5,hs1_10,hs1_15,hs1_20,hs1_25,hs1_30 = st.slider('5-min hotspot nodes', 0, 10, 1,key ='hs1_5'),st.slider('10-min hotspot nodes', 0, 10, 1,key ='hs1_10'),st.slider('15-min hotspot nodes', 0, 10, 2,key ='hs1_15'),st.slider('20-min hotspot nodes', 0, 10, 2,key ='hs1_20'),st.slider('25-min hotspot nodes', 0, 10, 2,key ='hs1_25'),st.slider('30-min hotspot nodes', 0, 10, 3,key ='hs1_30')      
            
            df1_x,df1_y = df1.iloc[0,50],df1.iloc[0,51]
            if ~df1_x.isdigit():
                df1_x,df1_y = 25,25            
            col1_center_x,col1_center_y = st.slider('center x coordinate', 0, 50, int(df1_x),key ='1x'),st.slider('center y coordinate', 0, 50, int(df1_y),key ='1y')

            col1_diameter = st.slider('diameter', 0, 50, 13,key ='1dmt')
            
            pixels = (2+col1_diameter)**2
            hs1_btm_30 = hs1_btm_25 = hs1_btm_20 = pixels-3 if pixels <= 200 else 200
            (top1,btm1) = ([0,hs1_5,hs1_10,hs1_15,hs1_20,hs1_25,hs1_30],[1,1,1,1,hs1_btm_20,hs1_btm_25,hs1_btm_30])
            
        with col2:
            drvt2_index_x_sets = []
            drvt2_index_y_sets = []
            ### IL6 sheet 15 anisotropy
            # List of options for the select box
            sheet_names2 = sheet_names
            selected_option2 = st.selectbox('Select an option', sheet_names2,key='sheet_names2')
            df2 = LOAD_DATA(inputDir=inputDir,dataDir=dataDir,sheet_name=selected_option2)
            
            hs2_5,hs2_10,hs2_15,hs2_20,hs2_25,hs2_30 = st.slider('5-min hotspot nodes', 0, 10, 0,key ='hs2_5'),st.slider('10-min hotspot nodes', 0, 10, 0,key ='hs2_10'),st.slider('15-min hotspot nodes', 0, 10,5,key ='hs2_15'),st.slider('20-min hotspot nodes', 0, 10, 10,key ='hs2_20'),st.slider('25-min hotspot nodes', 0, 10, 10,key ='hs2_25'),st.slider('30-min hotspot nodes', 0, 10, 10,key ='hs2_30')

            df2_x,df2_y = df2.iloc[0,50],df2.iloc[0,51]
            if ~df2_x.isdigit():
                df2_x,df2_y = 25,25            
            col2_center_x,col2_center_y = st.slider('center x coordinate', 0, 50, int(df2_x),key ='2x'),st.slider('center y coordinate', 0, 50, int(df2_y),key ='2y')

            col2_diameter = st.slider('diameter', 0, 50, 13,key ='2dmt')
            
            pixels = (2+col2_diameter)**2
            hs2_btm_30 = hs2_btm_25 = hs2_btm_20 = pixels-3 if pixels <= 200 else 200
            (top2,btm2) = ([0,hs2_5,hs2_10,hs2_15,hs2_20,hs2_25,hs2_30],[1,1,1,1,hs2_btm_20,hs2_btm_25,hs2_btm_30])
            
        submitted = st.form_submit_button("generate and compare!")              

    if submitted:
        col1_, col2_ = st.columns(2)
        with col1_:
            df=df1
            (x_start,x_end,y_start,y_end) = (col1_center_x-col1_diameter,col1_center_x+col1_diameter,col1_center_y-col1_diameter,col1_center_y+col1_diameter)
            for i in range(1,7,1):
                mtx = df.iloc[i*50:(i+1)*50,0:50]
                mtx = mtx.iloc[x_start:x_end,y_start:y_end] 
            
                if i == 0 | 1:
                    mtx_pre = pd.DataFrame(np.zeros(shape=(x_end-x_start, y_end-y_start)))
                    mtx_pre.columns = range(y_start,y_end)
                    mtx_pre.index=range(x_start,x_end)
                    (X,Y,Z)=plot_3D(mtx,previous = mtx_pre)
                    (x_drvtv,y_drvtv) = plotStream(mtx=mtx,top=top1[i],btm=btm1[i],previous = mtx_pre,
                                                                    x_unit = x_end-x_start,y_unit = y_end-y_start,
                                                                    cmlt_contour_color=cmlt_contour_color[i])
            
                else:
                    ## 10 mins interval
                    #mtx_pre = df.iloc[(i-2)*50:(i-1)*50,0:50] 
                    # 5 mins interval
                    mtx_pre = df.iloc[(i-1)*50:(i)*50,0:50] 
                    mtx_pre = mtx_pre.iloc[x_start:x_end,y_start:y_end]
                    (X,Y,Z) = plot_3D(mtx,previous = mtx_pre)
                    (x_drvtv,y_drvtv) = plotStream(mtx=mtx,top=top1[i],btm=btm1[i],previous = mtx_pre,
                                                                    x_unit = x_end-x_start,y_unit = y_end-y_start,
                                                                    cmlt_contour_color=cmlt_contour_color[i])
            
                ### Derivative index
                drvt1_index_x_sets.append(np.sum([np.abs(i) for i in np.array(x_drvtv[0:13]) - np.array(x_drvtv[-13:][::-1])])) 
                drvt1_index_y_sets.append(np.sum([np.abs(i) for i in np.array(y_drvtv[0:13]) - np.array(y_drvtv[-13:][::-1])])) 
           
        with col2_:      
            df=df2
            (x_start,x_end,y_start,y_end) = (col2_center_x-col2_diameter,col2_center_x+col2_diameter,col2_center_y-col2_diameter,col2_center_y+col2_diameter)
            for i in range(1,7,1):
                mtx = df.iloc[i*50:(i+1)*50,0:50]
                mtx = mtx.iloc[x_start:x_end,y_start:y_end] 
        
                if i == 0|1:
                    mtx_pre = pd.DataFrame(np.zeros(shape=(x_end-x_start, y_end-y_start)))
                    mtx_pre.columns = range(y_start,y_end)
                    mtx_pre.index=range(x_start,x_end)
                    (X,Y,Z)=plot_3D(mtx,previous = mtx_pre)
                    (x_drvtv,y_drvtv) = plotStream(mtx=mtx,top=top2[i],btm=btm2[i],previous = mtx_pre,
                                                                    x_unit = x_end-x_start,y_unit = y_end-y_start,
                                                                    cmlt_contour_color=cmlt_contour_color[i])
        
                else:
                    ## 10 mins interval
                    #mtx_pre = df.iloc[(i-2)*50:(i-1)*50,0:50] 
                    # 5 mins interval
                    mtx_pre = df.iloc[(i-1)*50:(i)*50,0:50]    
                    mtx_pre = mtx_pre.iloc[x_start:x_end,y_start:y_end]
                    (X,Y,Z) = plot_3D(mtx,previous = mtx_pre)
                    (x_drvtv,y_drvtv) = plotStream(mtx=mtx,top=top2[i],btm=btm2[i],previous = mtx_pre,
                                                                    x_unit = x_end-x_start,y_unit = y_end-y_start,
                                                                    cmlt_contour_color=cmlt_contour_color[i])
        
                ### Derivative index
                drvt2_index_x_sets.append(np.sum([np.abs(i) for i in np.array(x_drvtv[0:13]) - np.array(x_drvtv[-13:][::-1])])) 
                drvt2_index_y_sets.append(np.sum([np.abs(i) for i in np.array(y_drvtv[0:13]) - np.array(y_drvtv[-13:][::-1])])) 


def plot_time(IL6_21_IDI=[],IL6_15_IDI=[]):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(range(0,len(IL6_21_IDI),1),IL6_21_IDI,color='red',label="anisotropy")#np.repeat(0,len(IDI))
    ax.plot(range(0,len(IL6_15_IDI),1),IL6_15_IDI,color='blue',label="isotropy")#np.repeat(0,len(IDI))
    ax.set_xlabel('time')
    ax.set_ylabel('DII')
    ax.set_ylim(0,2.5)
    fig.show() 
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['5 mins','10 mins','15 mins','20 mins','25 mins','30 mins'])  
    plt.legend(loc="upper right",title='cell type')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Adjust the figure size
    fig.set_figwidth(5)  # Set the width in inches
    fig.set_figheight(5)  # Set the height in inches
    st.pyplot(fig)
    
step2 = st.checkbox('Step 2: show the derivative of inequality index (DII)',value=True)
if step2:
    DII1=np.sqrt(np.array(drvt1_index_x_sets)**2+np.array(drvt1_index_y_sets)**2)
    DII2=np.sqrt(np.array(drvt2_index_x_sets)**2+np.array(drvt2_index_y_sets)**2)
    _start = 0
    _end = 3
    col1__, col2__ = st.columns(2)
    with col1__:
        plot_time(IL6_21_IDI=DII1,IL6_15_IDI=DII2)
    with col2__:   
        table=pd.DataFrame({'anisotropy':DII1,'isotropy':DII2})
        try:
            table.index=['5 mins','10 mins','15 mins','20 mins','25 mins','30 mins']  
            st.write(table)
            st.markdown(get_table_download_link(table, fileName = "DII_comparison"), unsafe_allow_html=True)
        except:
            st.write("")
            
