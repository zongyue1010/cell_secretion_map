
# Contents of ~/my_app/streamlit_app.py
import streamlit as st
from PIL import Image

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
st.title('An online interactive analytical platform for cell secretion map')
st.header("Comparison of the cell secretion signals")


#############
### PAGES ###
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
import plotly.tools
import random
import json

from streamlit_cropper import st_cropper
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
#st.set_page_config(layout='wide')
from matplotlib import cm
import base64
from scipy import stats
import requests
from st_aggrid import GridUpdateMode, DataReturnMode
from plotly.graph_objs import Figure
from plotly.callbacks import Points, InputDeviceState
global value_counts

### color registration ###
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
    #print(xpos.shape)
    #print(ypos.shape)
    #print(dz_pre.shape)
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
    ax2.zaxis.set_major_locator(LinearLocator(6))
    ax2.zaxis.set_major_formatter(FormatStrFormatter('%d'))
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
    arr=np.insert(arr, 0, 0)
    diff = np.diff(arr)    
    # Divide by dx to approximate the derivative
    derivative = diff / dx
    return derivative

### plot inequality ###
def GINI_IDX(mtx=[],x_unit=[],y_unit=[]):
    ### lorenz curve ###
    x_val,y_val=mtx.sum(0),mtx.sum(1)
    x_inequ = np.array((x_val/x_val.sum()).cumsum()-(np.arange(0,1,1/x_unit)+1/x_unit))
    y_inequ = np.array((y_val/y_val.sum()).cumsum()-(np.arange(0,1,1/y_unit)+1/y_unit))    
    x_inequ=np.nan_to_num(x_inequ, copy=True, nan=0.0, posinf=None, neginf=None)
    y_inequ=np.nan_to_num(y_inequ, copy=True, nan=0.0, posinf=None, neginf=None)
    return(x_inequ,y_inequ)



### major function of plot stream ###
# https://stackoverflow.com/questions/16529892/adding-water-flow-arrows-to-matplotlib-contour-plot
from scipy.interpolate import Rbf
import io
import zipfile
def save_as_pdf(buffer,pdf_filename):
    # Create a PDF using reportlab
    
    c = canvas.Canvas(pdf_filename)
    img = io.BytesIO(buffer.getvalue())
    c.drawImage(img, 100, 500)  # Adjust the position and size as needed
    c.save()
    st.success(f'Successfully saved as {pdf_filename}')

def plotStream(mtx=[],lx=50,top=10,btm=10,**kwargs):
    # -- parameters --------------------
    lx,ly= np.size(mtx,0),np.size(mtx,1) # Work out matrix dimensions  
    cmlt_contour_color = kwargs['cmlt_contour_color'] if 'cmlt_contour_color' in kwargs.keys() else 'red'
    mtx_pre = kwargs['previous'] if 'previous' in kwargs.keys() else pd.DataFrame(np.zeros(shape=(50, 50)))
    x_unit = kwargs['x_unit'] if 'x_unit' in kwargs.keys() else 50 
    y_unit = kwargs['y_unit'] if 'y_unit' in kwargs.keys() else 50
    densityYN=kwargs['densityYN'] if 'densityYN' in kwargs.keys() else True
    density = kwargs['density'] if 'density' in kwargs.keys() else 1
    cntrlinecml = kwargs['cntrlinecml'] if 'cntrlinecml' in kwargs.keys() else 1200
    show_contour = kwargs['show_contour'] if 'show_contour' in kwargs.keys() else True
    show_cmlt_contour = kwargs['show_cmlt_contour'] if 'show_cmlt_contour' in kwargs.keys() else True
    timeline = kwargs['timeline'] if 'timeline' in kwargs.keys() else ''
    tick_labelsize = kwargs['tick_labelsize'] if 'tick_labelsize' in kwargs.keys() else 20
    dpi_value = kwargs['dpi_value'] if 'dpi_value' in kwargs.keys() else 300
    linewidth = kwargs['linewidth'] if 'linewidth' in kwargs.keys() else 6
    labelweight = kwargs['labelweight'] if 'labelweight' in kwargs.keys() else 'normal'
    framelinewidth = kwargs['framelinewidth'] if 'framelinewidth' in kwargs.keys() else 10
    fontname = kwargs['fontname'] if 'fontname' in kwargs.keys() else 'Arial'
    
    # -- Plot --------------------------
    # Set the font family for all text elements
    plt.rcParams['font.family'] = fontname    
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
    
    # get the positions of the masked cumulative expression
    shape = (lx,ly)
    mask_top_cmltv_pd = pd.DataFrame(mask_top_cmltv.reshape(shape))
    mask_top_cmltv_pd_pstn = np.where(mask_top_cmltv_pd == True)
    
    ### Plot flowlines cmltv arrow plot ###
    if densityYN == True:
        if top != 0:
            dy, dx = np.gradient(-zi_cmltv) # Flow goes down gradient (thus -zi)
            stream = ax.streamplot(xi[0,:], yi[:,0], dx, dy, density=1,arrowsize=1,linewidth=3,minlength=0.011,
                               color = 'grey',#broken_streamlines=True,
                               start_points=np.column_stack((mask_top_cmltv_pd_pstn[1],mask_top_cmltv_pd_pstn[0]))
                              )#color='0.6',
            # Customize the transparency level
            stream.lines.set_alpha(0.5)

        
    ### Contour gridded head observations ###
    contours =[]
    try:
        ax.contourf(xi, yi, zi, 
                          linewidths=2,levels=list(np.linspace(500,5000,7)),cmap=newcmp,#list(np.linspace(500,5000,7))
                            #cmap="jet2",
                          linestyles='dashed')  
        if show_cmlt_contour == True:
            contours = ax.contour(xi, yi, zi, 
                              linewidths=2,levels=list(np.linspace(500,5000,7)),cmap=newcmp,#list(np.linspace(500,5000,7))
                              linestyles='dashed')
            ax.clabel(contours)
    except:
        print("not available")          
    contours_cmltv = []
    if show_contour == True:
        try:
            contours_cmltv = ax.contour(xi, yi, zi_cmltv, [cntrlinecml], colors = cmlt_contour_color,alpha=0.5,linewidths=4) #
            ax.clabel(contours_cmltv,inline=True, fontsize=12)
        except:
            print("not available")     

    
    ####################
    ##### color ######
    #print(zi)
    #plt.imshow(zi,extent=[-lx, lx, -ly, ly], #cmap=REGISTER_TERRAIN_MAP(),#vmax=Bound, vmin=-Bound,, antialiased=False
    #                       interpolation='nearest',cmap='viridis')  # linewidth=1,
    
    
    ### Plot well locations ###
    #if top != 0:
    #    ax.plot(xpos[mask_top],ypos[mask_top], 'ko',color='black',markersize=2) #cmlt_contour_color
    #    #ax.plot(xpos[mask_top_cmltv],ypos[mask_top_cmltv], 'ko',color=)
    #    #ax.plot(xpos[mask_btm],ypos[mask_btm], 'ko',color='blue')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.invert_xaxis()
    ax.invert_yaxis()
    #ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    (x_inequ,y_inequ) = GINI_IDX(mtx=mtx,x_unit=x_unit,y_unit=y_unit)
    ### derivative of gini curve ###
    x_drvtv = list(DERIVATIVE(x_inequ,1))
    y_drvtv = list(DERIVATIVE(y_inequ,1))


    ### plot IDI (index of derivative inequality) ###
    # Set the maximum number of ticks on the x-axis
    max_ticks = 3
    import matplotlib.ticker as ticker
    #ax1.plot(np.arange(0,1,1/x_unit),(np.arange(0,1,1/x_unit)-np.arange(0,1,1/x_unit)),color='Black',linewidth=linewidth)  
    ax1.plot(np.arange(0,x_unit,1),(np.arange(0,x_unit,1)-np.arange(0,x_unit,1)),color='Black',linewidth=linewidth)  
    ax1.plot(np.arange(0,x_unit,1),x_drvtv,linewidth=linewidth)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.set_tick_params(labelsize=tick_labelsize) 
    # Set tick label font weight
    ax1.set_yticklabels(ax1.get_yticks(), rotation=0, weight=labelweight)    
    #ax1.yaxis.tick_right()
    
    # Set the new tick locations on the x-axis
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(max_ticks))
    #ax1.set_yticklabels([np.float(i) *100 for i in ax1.get_yticks()])
    #ax1.set_yticks(range(len([np.float(i)*100 for i in ax1.get_yticks()])))
    # Set tick label font weight
    # Optionally, you can format the tick labels if needed
    ax1.set_yticklabels(ax1.get_yticks(), rotation=0, weight=labelweight)    
    ax1.set_xticklabels(["{}".format(int(tick*100)) for tick in ax1.get_xticks()])
    ax1.set_yticklabels(["{}".format(int(tick*100)) for tick in ax1.get_yticks()]) 
    
    
    ax2.plot(np.arange(0,y_unit,1)-np.arange(0,y_unit,1),np.arange(0,y_unit,1),color='Black',linewidth=linewidth)
    ax2.plot(y_drvtv,np.arange(0,y_unit,1),linewidth=linewidth)
    ax2.invert_yaxis()
    ax2.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax2.yaxis.set_visible(False)
    # Set the new tick locations on the x-axis
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(max_ticks))
    #ax2.set_xticklabels([np.float(i)*100 for i in ax2.get_xticks()])
    #ax2.set_xticks(range(len([np.float(i)*100 for i in ax2.get_xticks()])))
    # Set tick label font weight
    # Optionally, you can format the tick labels if needed
    ax2.set_xticklabels(ax2.get_xticks(), rotation=0, weight=labelweight)
    ax2.set_xticklabels(["{}".format(int(tick*100)) for tick in ax2.get_xticks()])   
    ax2.set_yticklabels(["{}".format(int(tick*100)) for tick in ax2.get_yticks()])
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(framelinewidth)  # change width
        ax1.spines[axis].set_linewidth(framelinewidth)  # change width
        ax2.spines[axis].set_linewidth(framelinewidth)  # change width
        #ax.spines[axis].set_color('red')    # change color
    

    
    # Save the plot as png and pdf files
    plt.savefig('./output/'+timeline+".png",dpi=dpi_value) # , bbox_inches='tight'
    plt.savefig('./output/'+timeline+".pdf", format='pdf') # , bbox_inches='tight'

    # Adjust the margins (decrease them)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)
    
    
    ### save figure buffer ###
    buffer = io.BytesIO()
    plt.savefig(buffer, format='pdf')
    buffer.seek(0)
    ## Add a button to save as PDF
    #if st.button('Save as PDF '+timeline):
    #    save_as_pdf(buffer,timeline+'_figure.pdf')
    if st.download_button("Download PDF "+timeline, data=buffer, file_name=timeline+"_figure.pdf", mime="application/pdf"):
        st.success("File download initiated!")
    return(x_inequ,y_inequ,x_drvtv,y_drvtv,contours,contours_cmltv)  

def create_zip():
    with zipfile.ZipFile("figures.zip", "w") as zipf:
        for file in os.listdir("./output/"):
            if file.endswith(".png"):
                zipf.write("./output/"+file)
            if file.endswith(".pdf"):
                zipf.write("./output/"+file)

def get_contourline_df(contours):
    # Extract contour data
    contour_data = contours.collections[0].get_paths()
    
    # Get height values
    heights = contours.levels
    
    # Collect x, y, and height information
    data_points = []
    for path, height in zip(contour_data, heights):
        vertices = path.vertices
        for vertex in vertices:
            x_val, y_val = vertex
            data_points.append((x_val, y_val, height))
    
    # Create a pandas DataFrame
    df = pd.DataFrame(data_points, columns=['x', 'y', 'height'])
    
    # Display the DataFrame
    return(df)


### functions ###
### derivative ###
def DERIVATIVE(arr, dx):
    diff = np.diff(arr)    
    # Divide by dx to approximate the derivative
    derivative = np.array(diff / dx).round(3)
    return derivative

### plot inequality ###
def GINI_IDX(mtx=[],x_unit=[],y_unit=[]):
    ### lorenz curve ###
    x_val,y_val=mtx.sum(0),mtx.sum(1)
    x_inequ = np.array((x_val/x_val.sum()).cumsum()-(np.arange(0,1,1/x_unit)+1/x_unit))
    y_inequ = np.array((y_val/y_val.sum()).cumsum()-(np.arange(0,1,1/y_unit)+1/y_unit))    
    x_inequ=np.insert(x_inequ, 0, 0)
    y_inequ=np.insert(y_inequ, 0, 0)
    x_inequ=np.nan_to_num(x_inequ, copy=True, nan=0.0, posinf=None, neginf=None)
    y_inequ=np.nan_to_num(y_inequ, copy=True, nan=0.0, posinf=None, neginf=None)
    return(x_inequ,y_inequ)

### generate center weights ###
def generate_weights(x_drvtv):
    if len(x_drvtv)%2 == 0:
        radius = int(len(x_drvtv)/2)
        distance = np.array((np.linspace(1,radius,radius)))
        weight_vector = sqrt(1.0-(distance/np.max(distance))**2)
        weight_vector = np.concatenate((weight_vector[::-1],weight_vector))
    else:
        radius=int(len(x_drvtv)/2)+1
        distance = np.array((np.linspace(0,radius-1,radius)))
        weight_vector = sqrt(1.0-(distance/np.max(distance))**2)
        weight_vector = np.concatenate((weight_vector[::-1],weight_vector[1:]))
    return(weight_vector)

### generate center weighted signal-to-noise ratio ###
def generate_coverageIndex(x_drvtv):  
    ratio = len([x for x in x_drvtv if x > 0])/len([x for x in x_drvtv if x <= 0])
    weight=generate_weights(x_drvtv)
    
    x_drvtv= np.array(x_drvtv)
    binary = np.where(x_drvtv>0,x_drvtv,0)
    if sum(binary) != 0:
        signal=np.where(np.array(x_drvtv)>0,x_drvtv,0)
        #centralTendency=np.sum(signal*binary)
        centralTendency=np.sum(signal*ratio*weight)
    else:
        centralTendency = 0
    return(centralTendency)

def calculate_index(mtx,x_unit,y_unit):
    (x_inequ,y_inequ) = GINI_IDX(mtx=mtx,x_unit=x_unit,y_unit=y_unit)
    ### derivative of gini curve ###
    x_drvtv = list(DERIVATIVE(x_inequ,1))
    y_drvtv = list(DERIVATIVE(y_inequ,1))
    # index 1: DII degree of inequality index
    inequ = np.sqrt((x_inequ.mean()*2)**2 + (y_inequ.mean()*2)**2)
    # index 2: CWSNR center weighted signal-to-noise ratio
    #print(x_drvtv)
    coverageIndex = np.sqrt(generate_coverageIndex(x_drvtv)**2+generate_coverageIndex(y_drvtv)**2)
    return(inequ,coverageIndex)

def plot_time(IL6_21_IDI=[],IL6_15_IDI=[],xlabel = "",ylabel="",selected_option1='',selected_option2=''):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(range(0,len(IL6_21_IDI),1),IL6_21_IDI,color='red',label=selected_option1)#np.repeat(0,len(IDI))
    ax.plot(range(0,len(IL6_15_IDI),1),IL6_15_IDI,color='blue',label=selected_option2)#np.repeat(0,len(IDI))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_ylim(0,2.5)
    fig.show() 
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['5 mins','10 mins','15 mins','20 mins','25 mins','30 mins'])  
    plt.legend(loc="upper right",title='cell type')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Adjust the figure size
    fig.set_figwidth(5)  # Set the width in inches
    fig.set_figheight(5)  # Set the height in inches
    st.pyplot(fig)


def generate_colors(n):
    cmap = plt.get_cmap('tab10')  # Choose a colormap, e.g., 'tab10' which has 10 distinct colors
    colors = [cmap(i) for i in np.linspace(0, 1, n)]
    return colors
    
####################################################
### load the object and generate the coordincate ###
####################################################
@st.cache_data(ttl=60,max_entries=1,persist="disk")
def LOAD_DATA(inputDir='',dataDir='',sheet_name=''):
    df = pd.read_excel(r'./'+inputDir+dataDir, sheet_name=sheet_name,header=None)
    return df

###############################
### sidemenu of data source ###
###############################

def Real_world():
    st.header("Real-world Study")
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
    
    
    step1 = st.checkbox('Step 1: show the heatmap and delta changes of the cell signals',value=True)
    cmlt_contour_color = ['blue','blue','blue','orange','orange','red','red']
    
    ( drvt1_index_x_sets,  drvt1_index_y_sets) = ([],[])
    ( drvt2_index_x_sets,  drvt2_index_y_sets) = ([],[])
    ( selected_option1,  selected_option2) = ("","")

    ##############################
    ### font family suggestion ###
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    fontnames = sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))
    ##############################
        
    ####################
    ### main content ###
    ####################
    col1, col2 = st.columns(2)
    timeline=['0 mins','5 mins','10 mins','15 mins','20 mins','25 mins','30 mins']
    
    if step1:
        ##########################################
        ### general parameters for the figures ###
        ##########################################
        # pdi for png figure
        dpi_value = st.number_input("Enter a number for the figure pixel in the exported png file", min_value=1.0, max_value=1000.0, value=500.0, step=100.0,key ='dpi_value')
        # linewidth
        linewidth = st.number_input("Enter linewidth for the DII figure", min_value=1, max_value=20, value=10, step=1,key ='linewidth')   
        # frame line width
        framelinewidth= st.number_input("Enter frame linewidth for the figures", min_value=1, max_value=20, value=10, step=1,key ='framelinewidth')   
        # font family
        fontname = st.selectbox('Select a font style all the figure\'s labels', fontnames,key='fontnames')
        # tick_labelsize
        tick_labelsize = st.slider('tick label size',1, 80, 20,key ='tick_labelsize')    # 
        # labelweight
        labelweight = st.selectbox('Select an option for tick label style', ['bold','normal','heavy'],key='labelweight')    #     
        
        with st.form("form"):    
            with col1: 
                drvt1_index_x_sets = []
                drvt1_index_y_sets = []
                col1_inequ_set=[]
                col1_CWSNR_set=[]
                ### IL6 sheet 21 anisotropy
                # List of options for the select box
                #sheet_names1 = ['Sheet21-15max', 'Sheet20','Sheet19','Sheet18','Sheet17-15max','Sheet16','Sheet11','Sheet10-180c','Sheet9-180c',
                 #              'Sheet8-15max','Sheet7-15 max','Sheet6-depends','Sheet5-depends','Sheet3','Sheet2','Sheet1']
                ### load data ###
                sheet_names1 = sheet_names 
                selected_option1 = st.selectbox('Select an option', sheet_names1,key='sheet_names1')             
                df1 = LOAD_DATA(inputDir=inputDir,dataDir=dataDir,sheet_name=selected_option1)
                
                ### parameters ###
                # show node number set
                hs1_5,hs1_10,hs1_15,hs1_20,hs1_25,hs1_30 = st.slider('5-min hotspot nodes', 0, 10, 1,key ='hs1_5'),st.slider('10-min hotspot nodes', 0, 10, 1,key ='hs1_10'),st.slider('15-min hotspot nodes', 0, 10, 2,key ='hs1_15'),st.slider('20-min hotspot nodes', 0, 10, 2,key ='hs1_20'),st.slider('25-min hotspot nodes', 0, 10, 2,key ='hs1_25'),st.slider('30-min hotspot nodes', 0, 10, 3,key ='hs1_30')
                # show arrow density
                densityYN1 = st.checkbox('Show arrow',value=False,key ='densityYN1')
                density1 = st.slider('arrow density', 0.0, 10.0, 1.0,key ='density1') 
    
                # countour line 
                show_contour1 = st.checkbox('Show signal contour line',value=False,key ='show_contour1')         
                
                # countour line of cumulative
                show_cmlt_contour1 = st.checkbox('Show cumulative signal contour line',value=False,key ='show_cmlt_contour1')
                cntrlinecml1 = st.number_input("Enter a number for the cumulative signal contour line", min_value=0.0, max_value=5000.0, value=1200.0, step=50.0,key ='cntrlinecml1')
                       
                # center coordinate
                if df1.shape[1]>50:
                    df1_x,df1_y = df1.iloc[0,50],df1.iloc[0,51]
                    ## Correct usage by converting the integer to a string
                    number_str = str(df1_x)
                    if ~number_str.isdigit():
                        df1_x,df1_y = df1.iloc[1,50],df1.iloc[1,51]       
                else:
                    df1_x,df1_y = 25,25
                    
                col1_center_x,col1_center_y = st.slider('center x coordinate', 0, 50, int(df1_x),key ='1x'),st.slider('center y coordinate', 0, 50, int(df1_y),key ='1y') 

                min_1=np.int32(np.min([df1_x, 50-df1_x,df1_y,50-df1_y]))
                # apothem
                col1_apothem = st.slider('apothem', 0, min_1, min_1,key ='1dmt')          
                pixels = (2+col1_apothem)**2
                hs1_btm_30 = hs1_btm_25 = hs1_btm_20 = pixels-3 if pixels <= 200 else 200
                (top1,btm1) = ([0,hs1_5,hs1_10,hs1_15,hs1_20,hs1_25,hs1_30],[1,1,1,1,hs1_btm_20,hs1_btm_25,hs1_btm_30])
                
     
            with col2:
                drvt2_index_x_sets = []
                drvt2_index_y_sets = []
                col2_inequ_set=[]
                col2_CWSNR_set=[]
                ### IL6 sheet 15 anisotropy
                # List of options for the select box
                sheet_names2 = sheet_names
                selected_option2 = st.selectbox('Select an option', sheet_names2,key='sheet_names2')
                df2 = LOAD_DATA(inputDir=inputDir,dataDir=dataDir,sheet_name=selected_option2)
                
                ### parameters ###
                # show node number set
                hs2_5,hs2_10,hs2_15,hs2_20,hs2_25,hs2_30 = st.slider('5-min hotspot nodes', 0, 10, 1,key ='hs2_5'),st.slider('10-min hotspot nodes', 0, 10, 1,key ='hs2_10'),st.slider('15-min hotspot nodes', 0, 10, 2,key ='hs2_15'),st.slider('20-min hotspot nodes', 0, 10, 2,key ='hs2_20'),st.slider('25-min hotspot nodes', 0, 10, 2,key ='hs2_25'),st.slider('30-min hotspot nodes', 0, 10, 3,key ='hs2_30')
                # show arrow density
                densityYN2 = st.checkbox('Show arrow',value=False,key ='densityYN2')
                density2 = st.slider('arrow density', 0.0, 10.0, 1.0,key ='density2')
                
                # countour line 
                show_contour2 = st.checkbox('Show signal contour line',value=False,key ='show_contour2')            
                # countour line of culmulative
                show_cmlt_contour2 = st.checkbox('Show cumulative signal contour line',value=False,key ='show_cmlt_contour2')
    
                cntrlinecml2 = st.number_input("Enter a number for the cumulative signal contour line", min_value=0.0, max_value=5000.0, value=1200.0, step=50.0,key ='cntrlinecml2')
                
                # center coordinate
                if df2.shape[1]>50:
                    df2_x,df2_y = df2.iloc[0,50],df2.iloc[0,51]
                    ## Correct usage by converting the integer to a string
                    number_str = str(df2_x)                
                    if ~number_str.isdigit():
                        df2_x,df2_y = df2.iloc[1,50],df2.iloc[1,51]       
                else:
                    df2_x,df2_y = 25,25
                    
                col2_center_x,col2_center_y = st.slider('center x coordinate', 0, 50, int(df2_x),key ='2x'),st.slider('center y coordinate', 0, 50, int(df2_y),key ='2y')
                min_2=np.int32(np.min([df2_x, 50-df2_x,df2_y,50-df2_y]))
                col2_apothem = st.slider('apothem', 0, min_2, min_2,key ='2dmt')           
                pixels = (2+col2_apothem)**2
                hs2_btm_30 = hs2_btm_25 = hs2_btm_20 = pixels-3 if pixels <= 200 else 200
                (top2,btm2) = ([0,hs2_5,hs2_10,hs2_15,hs2_20,hs2_25,hs2_30],[1,1,1,1,hs2_btm_20,hs2_btm_25,hs2_btm_30])
                
            submitted = st.form_submit_button("Generate and compare!")              
    
        if submitted:
    
            col1_, col2_ = st.columns(2)
            with col1_:
                contours_dfs = pd.DataFrame()
                contours_cmltv_dfs = pd.DataFrame()
                df=df1
                (x_start,x_end,y_start,y_end) = (col1_center_x-col1_apothem,col1_center_x+col1_apothem,col1_center_y-col1_apothem,col1_center_y+col1_apothem)
                for i in range(1,7,1):
                    mtx = df.iloc[i*50:(i+1)*50,0:50]
                    mtx = mtx.iloc[x_start:x_end,y_start:y_end] 
                
                    if (i == 0) or (i == 1):
                        mtx_pre = pd.DataFrame(np.zeros(shape=(x_end-x_start, y_end-y_start)))
                        mtx_pre.columns = range(y_start,y_end)
                        mtx_pre.index=range(x_start,x_end)
                        (X,Y,Z)=plot_3D(mtx,previous = mtx_pre)
                        (x_inequ,y_inequ,x_drvtv,y_drvtv,contours,contours_cmltv) = plotStream(
                            mtx=mtx,top=top1[i],btm=btm1[i],previous = mtx_pre,
                            x_unit = x_end-x_start,y_unit = y_end-y_start,
                            cmlt_contour_color=cmlt_contour_color[i],
                            densityYN=densityYN1,density=density1,
                            show_contour=show_cmlt_contour1,show_cmlt_contour=show_cmlt_contour1,
                            cntrlinecml=cntrlinecml1,
                            timeline = 'a_'+timeline[i],
                            tick_labelsize = tick_labelsize,
                            dpi_value = dpi_value,
                            linewidth = linewidth,
                            labelweight=labelweight,
                            framelinewidth=framelinewidth,
                            fontname=fontname
                        )
                
                    else:
                        ## 10 mins interval
                        #mtx_pre = df.iloc[(i-2)*50:(i-1)*50,0:50] 
                        # 5 mins interval
                        mtx_pre = df.iloc[(i-1)*50:(i)*50,0:50] 
                        mtx_pre = mtx_pre.iloc[x_start:x_end,y_start:y_end]
                        (X,Y,Z) = plot_3D(mtx,previous = mtx_pre)
                        (x_inequ,y_inequ,x_drvtv,y_drvtv,contours,contours_cmltv) = plotStream(
                            mtx=mtx,top=top1[i],btm=btm1[i],previous = mtx_pre,
                            x_unit = x_end-x_start,y_unit = y_end-y_start,
                            cmlt_contour_color=cmlt_contour_color[i],
                            densityYN=densityYN1,density=density1,
                            show_contour=show_cmlt_contour1,show_cmlt_contour=show_cmlt_contour1,    
                            cntrlinecml=cntrlinecml1,
                            timeline = 'a_'+timeline[i],
                            tick_labelsize = tick_labelsize,
                            dpi_value = dpi_value,
                            linewidth = linewidth,
                            labelweight=labelweight,
                            framelinewidth=framelinewidth,
                            fontname=fontname
                        )
                    mtx = mtx.reset_index(drop=True)
                    mtx_pre = mtx_pre.reset_index(drop=True)
                    mtx_delta=mtx-mtx_pre
                
                    ### Derivative index
                    drvt1_index_x_sets.append(np.sum([np.abs(i) for i in np.array(x_drvtv[0:col1_apothem]) - np.array(x_drvtv[-col1_apothem:][::-1])])) 
                    drvt1_index_y_sets.append(np.sum([np.abs(i) for i in np.array(y_drvtv[0:col1_apothem]) - np.array(y_drvtv[-col1_apothem:][::-1])])) 
    
    
                    ### contour line export
                    if contours:
                        contours_df = get_contourline_df(contours)
                        contours_df['time'] = timeline[i]
                        contours_dfs = pd.concat([contours_dfs,contours_df])
                    if contours_cmltv:
                        contours_cmltv_df = get_contourline_df(contours_cmltv)
                        contours_cmltv_df['time'] = timeline[i]
                        contours_cmltv_dfs = pd.concat([contours_cmltv_dfs,contours_cmltv_df])                   
    
                    ### calculate 
                    inequ,CWSNR=calculate_index(mtx_delta,col1_apothem*2,col1_apothem*2)
                    col1_inequ_set.append(inequ)
                    col1_CWSNR_set.append(CWSNR) 
                
                st.markdown(get_table_download_link(contours_dfs, fileName = "contour_line.txt"), unsafe_allow_html=True)
                st.markdown(get_table_download_link(contours_cmltv_dfs, fileName = "cumulative_contour_line.txt"), unsafe_allow_html=True)
                  
                    
            with col2_:      
                contours_dfs = pd.DataFrame()
                contours_cmltv_dfs = pd.DataFrame()
    
                
                df=df2
                (x_start,x_end,y_start,y_end) = (col2_center_x-col2_apothem,col2_center_x+col2_apothem,col2_center_y-col2_apothem,col2_center_y+col2_apothem)
                for i in range(1,7,1):
                    mtx = df.iloc[i*50:(i+1)*50,0:50]
                    mtx = mtx.iloc[x_start:x_end,y_start:y_end] 
            
                    if (i == 0) or (i == 1):
                        mtx_pre = pd.DataFrame(np.zeros(shape=(x_end-x_start, y_end-y_start)))
                        mtx_pre.columns = range(y_start,y_end)
                        mtx_pre.index=range(x_start,x_end)
                        (X,Y,Z)=plot_3D(mtx,previous = mtx_pre)
                        (x_inequ,y_inequ,x_drvtv,y_drvtv,contours,contours_cmltv) = plotStream(
                            mtx=mtx,top=top2[i],btm=btm2[i],previous = mtx_pre,
                            x_unit = x_end-x_start,y_unit = y_end-y_start,
                            cmlt_contour_color=cmlt_contour_color[i],
                            densityYN=densityYN2,density=density2,
                            show_contour=show_cmlt_contour2,show_cmlt_contour=show_cmlt_contour2,    
                            cntrlinecml=cntrlinecml2,
                            timeline = 'b_'+timeline[i],
                            tick_labelsize = tick_labelsize,
                            dpi_value = dpi_value,
                            linewidth = linewidth,
                            labelweight=labelweight,
                            framelinewidth=framelinewidth,
                            fontname=fontname
                        )
            
                    else:
                        ## 10 mins interval
                        #mtx_pre = df.iloc[(i-2)*50:(i-1)*50,0:50] 
                        # 5 mins interval
                        mtx_pre = df.iloc[(i-1)*50:(i)*50,0:50]    
                        mtx_pre = mtx_pre.iloc[x_start:x_end,y_start:y_end]
                        (X,Y,Z) = plot_3D(mtx,previous = mtx_pre)
                        (x_inequ,y_inequ,x_drvtv,y_drvtv,contours,contours_cmltv) = plotStream(
                            mtx=mtx,top=top2[i],btm=btm2[i],previous = mtx_pre,
                            x_unit = x_end-x_start,y_unit = y_end-y_start,
                            cmlt_contour_color=cmlt_contour_color[i],
                            densityYN=densityYN2,density=density2,
                            show_contour=show_cmlt_contour2,show_cmlt_contour=show_cmlt_contour2, 
                            cntrlinecml=cntrlinecml2,
                            timeline = 'b_'+timeline[i],
                            tick_labelsize = tick_labelsize,
                            dpi_value = dpi_value,
                            linewidth = linewidth,
                            labelweight=labelweight,
                            framelinewidth=framelinewidth,
                            fontname=fontname
                            
                        )
                    mtx = mtx.reset_index(drop=True)
                    mtx_pre = mtx_pre.reset_index(drop=True)
                    mtx_delta=mtx-mtx_pre
                    
                    ### Derivative index
                    drvt2_index_x_sets.append(np.sum([np.abs(i) for i in np.array(x_drvtv[0:col2_apothem]) - np.array(x_drvtv[-col2_apothem:][::-1])])) 
                    drvt2_index_y_sets.append(np.sum([np.abs(i) for i in np.array(y_drvtv[0:col2_apothem]) - np.array(y_drvtv[-col2_apothem:][::-1])])) 
                    
                    ### contour line export
                    if contours:
                        contours_df = get_contourline_df(contours)
                        contours_df['time'] = timeline[i]
                        contours_dfs = pd.concat([contours_dfs,contours_df])
                    if contours_cmltv:
                        contours_cmltv_df = get_contourline_df(contours_cmltv)
                        contours_cmltv_df['time'] = timeline[i]
                        contours_cmltv_dfs = pd.concat([contours_cmltv_dfs,contours_cmltv_df])                
     
                    ### calculate 
                    inequ,CWSNR=calculate_index(mtx_delta,col2_apothem*2,col2_apothem*2)
                    col2_inequ_set.append(inequ)
                    col2_CWSNR_set.append(CWSNR) 
                    
                st.markdown(get_table_download_link(contours_dfs, fileName = "contour_line.txt"), unsafe_allow_html=True)
                st.markdown(get_table_download_link(contours_cmltv_dfs, fileName = "cumulative_contour_line.txt"), unsafe_allow_html=True)
        ### zip files and download ###         
        create_zip()             
        # Provide download link
        with open("figures.zip", "rb") as f:
            st.download_button("Download Figures", f.read(), file_name="figures.zip")
            
        
    step2 = st.checkbox('Step 2: show the measurement of the dissymmetry and diffusion of cumulative signals using Signal Inequality Index (SII) and Singal Coverage Index (SCI)',value=True)
    if step2:
        DII1=np.sqrt(np.array(drvt1_index_x_sets)**2+np.array(drvt1_index_y_sets)**2)
        DII2=np.sqrt(np.array(drvt2_index_x_sets)**2+np.array(drvt2_index_y_sets)**2)
        _start = 0
        _end = 3
        col1__, col2__ = st.columns(2)
        with col1__:
            #plot_time(IL6_21_IDI=DII1,IL6_15_IDI=DII2,xlabel='time',ylabel="DII")
            plot_time(IL6_21_IDI=col1_inequ_set,IL6_15_IDI=col2_inequ_set,xlabel='time',ylabel="SII",
                      selected_option1=selected_option1,selected_option2=selected_option2)
        with col2__:
            plot_time(IL6_21_IDI=col1_CWSNR_set,IL6_15_IDI=col2_CWSNR_set,xlabel='time',ylabel="SCI",
                     selected_option1=selected_option1,selected_option2=selected_option2)
          
        #table=pd.DataFrame({selected_option1:DII1,selected_option2:DII2})
        table_inequ=pd.DataFrame({selected_option1+'_SII':col1_inequ_set,selected_option2+'_SII':col2_inequ_set})
        table_CWSNR=pd.DataFrame({selected_option1+'_SCI':col1_CWSNR_set,selected_option2+'_SCI':col2_CWSNR_set})
        try:
            #table.index=['5 mins','10 mins','15 mins','20 mins','25 mins','30 mins']  
            #st.write(table)
            #st.markdown(get_table_download_link(table, fileName = "DII_comparison"), unsafe_allow_html=True)
            table_inequ.index=['5 mins','10 mins','15 mins','20 mins','25 mins','30 mins']            
            table_CWSNR.index=['5 mins','10 mins','15 mins','20 mins','25 mins','30 mins']            
            table_cmb = pd.merge(table_inequ,table_CWSNR,right_index=True,left_index= True)
            st.write(table_cmb)
            st.markdown(get_table_download_link(table_cmb, fileName = "SII_SCI_metrics"), unsafe_allow_html=True)
                    
        except:
            st.write("")
            
def upload():
    st.markdown("## Data Upload")
    # Upload the dataset and save as csv
    st.markdown("### Upload a csv file for analysis.") 
    st.write("\n")

    # Code to read a single file 
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx', 'txt'])
    data = pd.DataFrame()
    if uploaded_file is not None:
        #st.write(uploaded_file.name)
        if len(re.findall("\.txt",uploaded_file.name))>0:
            data = pd.read_csv(uploaded_file,sep="\t",header=None)
        elif len(re.findall("\.csv",uploaded_file.name))>0:
            data = pd.read_csv(uploaded_file,header=None)
        else:
            data = pd.read_excel(uploaded_file,header=None)
    drvt_index_x_sets = []
    drvt_index_y_sets = []
    col_inequ_set=[]
    col_CWSNR_set=[]    
    
    (drvt_index_x_sets,  drvt_index_y_sets) = ([],[])
    (selected_option) = ("")
    
    st.session_state['data'] = pd.DataFrame() if 'data' not in st.session_state.keys() else st.session_state['data']
    ''' Load the data and save the columns with categories as a dataframe. 
    This section also allows changes in the numerical and categorical columns. '''
    if st.button("Load Data") or st.session_state['data'].shape[0]>0:   
        if data.shape[0]>0:
            # Raw data 
            st.dataframe(data)
            st.session_state['data'] = data
            t_time = np.int32(data.shape[0]/data.shape[1])
            cmlt_contour_color = generate_colors(t_time)
             
            
            with st.form("form_"):
                df = st.session_state['data']
                fontnames = sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))
                ### parameters ###
                # pdi for png figure
                dpi_value = st.number_input("Enter a number for the figure pixel in the exported png file", min_value=1.0, max_value=1000.0, value=500.0, step=100.0,key ='dpi_value')
                # linewidth
                linewidth = st.number_input("Enter linewidth for the DII figure", min_value=1, max_value=20, value=10, step=1,key ='linewidth')   
                # frame line width
                framelinewidth= st.number_input("Enter frame linewidth for the figures", min_value=1, max_value=20, value=10, step=1,key ='framelinewidth')   
                # font family
                fontname = st.selectbox('Select a font style all the figure\'s labels', fontnames,key='fontnames')
                # tick_labelsize
                tick_labelsize = st.slider('tick label size',1, 80, 20,key ='tick_labelsize')    #            
                # labelweight
                labelweight = st.selectbox('Select an option for tick label style', ['bold','normal','heavy'],key='labelweight')    #           
        
                ### parameters ###
                hs_set = []
                hs_btm_set = []
                
                # show node number set
                for t_idx in range(0,t_time,1):
                    hs_ = st.slider('hotspot nodes at timepoint '+str(t_idx), 0, 10, 1,key ='hs_'+str(t_idx))
                    hs_set.append(hs_)
                # show arrow density
                densityYN = st.checkbox('Show arrow',value=False,key ='densityYN')
                density = st.slider('arrow density', 0.0, 10.0, 1.0,key ='density') 
                
                # countour line 
                show_contour = st.checkbox('Show signal contour line',value=False,key ='show_contour')         
                
                # countour line of cumulative
                show_cmlt_contour = st.checkbox('Show cumulative signal contour line',value=False,key ='show_cmlt_contour')
                cntrlinecml = st.number_input("Enter a number for the cumulative signal contour line", min_value=0.0, max_value=5000.0, value=1200.0, step=50.0,key ='cntrlinecml')
                       
                # center coordinate
                df_x = np.int32(df.shape[1]/2)
                df_y = np.int32(df.shape[1]/2)
                
                col_center_x,col_center_y = st.slider('center x coordinate', 0, df.shape[1], np.int32(df_x),key ='1x'),st.slider('center y coordinate', 0, df.shape[1], np.int32(df_y),key ='1y')

                min_=np.int32(np.min([df_x, df.shape[1]-df_x,df_y,df.shape[1]-df_y]))
                # apothem 
                col_apothem = st.slider('apothem', 0, min_, min_,key ='dmt')          
                pixels = (2+col_apothem)**2
                
                for t_idx in range(0,t_time,1):
                    hs_btm = pixels-3 if pixels <= 200 else 200
                    hs_btm_set.append(hs_btm)
                (top1,btm1) = ([hs_],[hs_btm])      
                submitted = st.form_submit_button("Generate!")
                
            if submitted:
                contours_dfs = pd.DataFrame()
                contours_cmltv_dfs = pd.DataFrame()
                (x_start,x_end,y_start,y_end) = (col_center_x-col_apothem,col_center_x+col_apothem,col_center_y-col_apothem,col_center_y+col_apothem)  
                rangeVal = df.shape[1]
    
    
                
                for t in range(0,t_time,1):
                    #st.write(t)
                    mtx = df.iloc[t*rangeVal:(t+1)*rangeVal,0:rangeVal]
                    mtx = mtx.iloc[x_start:x_end,y_start:y_end]       
                    
                    if (t == 0) or (t == 1): 
                        #st.write('Yes')
                        mtx_pre = pd.DataFrame(np.zeros(shape=(col_apothem*2, col_apothem*2)))
                        mtx_pre.columns = list(range(y_start,y_end))
                        mtx_pre.index=list(range(x_start,x_end))
                        (X,Y,Z)=plot_3D(mtx,previous = mtx_pre)
                        (x_inequ,y_inequ,x_drvtv,y_drvtv,contours,contours_cmltv) = plotStream(
                            mtx=mtx,top=hs_set[t],btm=hs_btm_set[t],previous = mtx_pre,
                            x_unit = x_end-x_start,y_unit = y_end-y_start,
                            cmlt_contour_color=cmlt_contour_color[t],
                            densityYN=densityYN,density=density,
                            show_contour=show_cmlt_contour,show_cmlt_contour=show_cmlt_contour,
                            cntrlinecml=cntrlinecml,
                            timeline = str(t),
                            tick_labelsize = tick_labelsize,
                            dpi_value = dpi_value,
                            linewidth = linewidth,
                            labelweight=labelweight,
                            framelinewidth=framelinewidth,
                            fontname=fontname
                        )
                    else:
                        #st.write(t)
                        ## 10 mins interval
                        #mtx_pre = df.iloc[(i-2)*50:(i-1)*50,0:50] 
                        # 5 mins interval          
                        mtx_pre = df.iloc[(t-1)*rangeVal:(t)*rangeVal,0:rangeVal]    
                        mtx_pre = mtx_pre.iloc[x_start:x_end,y_start:y_end]                  
                        (X,Y,Z) = plot_3D(mtx,previous = mtx_pre)
                        (x_inequ,y_inequ,x_drvtv,y_drvtv,contours,contours_cmltv) = plotStream(
                            mtx=mtx,top=hs_set[t],btm=hs_btm_set[t],previous = mtx_pre,
                            x_unit = x_end-x_start,y_unit = y_end-y_start,
                            cmlt_contour_color=cmlt_contour_color[t],
                            densityYN=densityYN,density=density,
                            show_contour=show_cmlt_contour,show_cmlt_contour=show_cmlt_contour, 
                            cntrlinecml=cntrlinecml,
                            timeline = str(t),
                            tick_labelsize = tick_labelsize,
                            dpi_value = dpi_value,
                            linewidth = linewidth,
                            labelweight=labelweight,
                            framelinewidth=framelinewidth,
                            fontname=fontname   
                        )
                    mtx = mtx.reset_index(drop=True)
                    mtx_pre = mtx_pre.reset_index(drop=True)
                    mtx_delta=mtx-mtx_pre
                    ### Derivative index
                    drvt_index_x_sets.append(np.sum([np.abs(i) for i in np.array(x_drvtv[0:col_apothem]) - np.array(x_drvtv[-col_apothem:][::-1])])) 
                    drvt_index_y_sets.append(np.sum([np.abs(i) for i in np.array(y_drvtv[0:col_apothem]) - np.array(y_drvtv[-col_apothem:][::-1])])) 
                       
            
                    ### contour line export
                    if contours:
                        contours_df = get_contourline_df(contours)
                        contours_df['time'] = ''#timeline[i]
                        contours_dfs = pd.concat([contours_dfs,contours_df])
                    if contours_cmltv:
                        contours_cmltv_df = get_contourline_df(contours_cmltv)
                        contours_cmltv_df['time'] = ''#timeline[i]
                        contours_cmltv_dfs = pd.concat([contours_cmltv_dfs,contours_cmltv_df])                   
                    ### calculate 
                    inequ,CWSNR=calculate_index(mtx_delta,col_apothem*2,col_apothem*2)
                    col_inequ_set.append(inequ)
                    col_CWSNR_set.append(CWSNR)   
                    st.markdown(get_table_download_link(contours_dfs, fileName = "contour_line.txt"), unsafe_allow_html=True)
                    st.markdown(get_table_download_link(contours_cmltv_dfs, fileName = "cumulative_contour_line.txt"), unsafe_allow_html=True)
    #st.write(col_inequ_set)
    #st.write(col_CWSNR_set)
    try:
        table_inequ=pd.DataFrame({'SII':col_inequ_set})
        table_CWSNR=pd.DataFrame({'SCI':col_CWSNR_set})     
        table_inequ.index=list(range(0,t_time,1))
        table_CWSNR.index=list(range(0,t_time,1))
        table_ = pd.merge(table_inequ,table_CWSNR,right_index=True,left_index= True)
        st.write(table_)
        st.markdown(get_table_download_link(table_, fileName = "SII_SCI_metrics"), unsafe_allow_html=True)
                  
    except:
        st.write("")            
###################
### select menu ###
###################
page_names_to_funcs = {
    "Real-world studies": Real_world,
    "Upload your data": upload,
}

selected_page = st.sidebar.selectbox("Section", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


##############
### footer ###
##############
st.header('Cite us:')

st.markdown(f"\n*Zongliang Yue\*, Lang Zhou, Fengyuan Huang and Pengyu Chen\**, S2Map: An online interactive analytical platform for cell secretion map generation, under review.")
st.header('About us:')
st.write(f"If you have questions or comments about the database contents or technical support, please email Dr. Zongliang Yue, zzy0065@auburn.edu")
st.write("Our Research group: AI.pharm, Auburn University, Auburn, USA. https://github.com/ai-pharm-AU")
### end ###
###########