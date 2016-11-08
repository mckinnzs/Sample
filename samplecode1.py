# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:14:57 2016

@author: Zachery McKinnon
"""

import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file

def read_and_join(folder_path, index):
    """Takes a folder of csv's with a common index column and 
    returns a single pandas dataframe with the data from all csv's joined. 
    Column names are renamed based on order of file read 
    (e.g. columns in the first file read receive a suffix of _f1)"""
    
    #reads csv and places them in dictionary, filling NaN entries with 0
    all_files = glob.glob(folder_path + "/*.csv")
    df_dict = {}
    key = 1
    for file_ in all_files:
       file_df = pd.read_csv(file_, index_col=index, header=0)
       df_dict[key] = file_df.fillna(0)
       key += 1

    #renames columns based on source file
    df = None
    for suffix, input_df in df_dict.items():
        renamer_dict = {}
        for col in list(input_df.columns):
            renamer_dict[col] = col + "_f" + str(suffix)
        input_df.rename(columns=renamer_dict, inplace=True)

        #joins columns (Note that how = 'inner' will return only the rows with 
        #an index key in all of the csv files, whereas 'outer' will
        #return all rows and add NaN for missing values)
        if df is None:
            df = input_df
        else:
            df = df.join(input_df, how='inner')       
    return df
            
def time_split(pandas_df, column_name):
    """Takes a pandas_df with column_name of 
    form 0 days 00:00:00.000000000 and creates a pandas df
    with two columns, days and remaining seconds"""
    time = pd.DataFrame(pandas_df[column_name].str.split(' days ').tolist(), 
                    columns = ['days','stamp'])
    days_reading = time['days'].astype(float)
    subdays_reading = time['stamp']
    seconds_reading = []
    for row in subdays_reading:
        row = row[:-3]
        t = datetime.strptime(row,"%H:%M:%S.%f")
        delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, 
                      microseconds = t.microsecond).total_seconds()
        seconds_reading.append(delta)
    time_df = pd.DataFrame({'days': days_reading, 'seconds': seconds_reading})
    return time_df
      
def PCA_and_kmeans(pandas_df, clusters):
    """Takes a pandas df and plots the results of kmeans++ (with specified 
    clusters) on a PCA-reduced 2D graph """

    #converts pandas df to an np array with no objects          
    np_array = np.array(pandas_df.select_dtypes(exclude=['object']))   
                                                            
    #implements PCA and Kmeans
    reduced_data = PCA(n_components=2).fit_transform(np_array)
    kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
    kmeans.fit(reduced_data)
    
    #plots results
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 50), np.arange(y_min, y_max, 50))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    
    #formats and plots cluster centroids
    ax = plt.subplot()
    ax.spines["top"].set_color('gray')    
    ax.spines["right"].set_color('gray')  
    ax.spines["left"].set_color('gray') 
    ax.spines["bottom"].set_color('gray') 
    
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=100, color='b')
    plt.title('K-means++ clustering (PCA-reduced data)\n')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    
def histogram(pandas_df, column_name, bins):
    """Creates a histogram of the column_name of a pandas_df 
    with a specified number of bins."""
    
    #sets minimum and maximum x values rounded to 10 for the plot
    x_min = int(pandas_df[column_name].min()) - int(pandas_df[column_name].min())%10
    x_max = int(pandas_df[column_name].max()) + 10 - int(pandas_df[column_name].max())%10
    
    #plots histogram
    plt.figure(figsize=(12, 9))
    n, _, _ = plt.hist(pandas_df[column_name], alpha = 0.5, bins = bins,
                       range = [x_min, x_max])
    
    #determines max y value rounded to 10 for the plot
    y_max = int(n.max() + 10 - n.max()%10)
    
    #formats the plot
    ax = plt.subplot()
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False) 
    ax.set_ylim([0, y_max])
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    if y_max <50:
        for y in range(0, y_max+1, 5):    
            plt.plot(range(x_min,x_max+1), [y] * len(range(x_min,x_max+1)),
                     "--", lw=0.5, color="black", alpha=0.3)
    else:
         for y in range(0, y_max+1, 10):    
            plt.plot(range(x_min,x_max+1), [y] * len(range(x_min,x_max+1)),
                     "--", lw=0.5, color="black", alpha=0.3)
    plt.xlabel(column_name, fontsize=16)
    plt.ylabel("Count", fontsize=16)

def circle_graph(pandas_df, index, column_list, unit):
    """Takes a pandas df with an index column and a list of columns (3 to 5 for 
    best results) with float values representing the unit and creates a circle graph"""
    
    """eg circle_graph(survivor_df, 'days lasted', ['15 to 24 y.o.', '25 to 34 y.o.', '35+ y.o.'], '%')"""
    
        #Set the colors of the bars in the bar graph based on "Tableau 20" colors.    
    tableau20 = [ '#1f77b4',  '#2ca02c','#7f7f7f','#8c564b','#d62728', '#bcbd22', '#9edae5','#17becf',
                 '#98df8a','#9467bd','#aec7e8','#c5b0d5', '#c7c7c7']
    width = 800
    height = 800
    inner_radius = 90
    outer_radius = 290
    big_angle = 2.0 * np.pi / (len(pandas_df) + 1)
    small_angle = big_angle / (len(column_list) *2 + 1)
    p = figure(plot_width=width, plot_height=height, title="", x_axis_type=None, y_axis_type=None,
    x_range=(-420, 420), y_range=(-420, 420), min_border=0, outline_line_color="white")
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    #draw the large wedges
    angles = np.pi/2 - big_angle/2 - pandas_df.index.to_series()*big_angle
    p.annular_wedge(0, 0, inner_radius, outer_radius, -big_angle + angles, angles, color='#cfcfcf',)
    
    #find the maximum value for the concentric rings of values
    column_max = list()
    for column in column_list:
        column_max.append(max(pandas_df[column]))
    label_max = int(max(column_max) + 10 - max(column_max)%10)
    
    #draw the small wedges and labels
    bar_color = {}
    counter = 0
    for column in column_list:
        bar_color[column] = tableau20[counter]
        p.annular_wedge(0, 0, inner_radius, 90 + pandas_df[column]*(200/float(label_max)),
                -big_angle+angles+(2*counter+1)*small_angle, -big_angle+angles+(2*counter+2)*small_angle,
                color=bar_color[column])
        p.rect([-40, -40, -40], [37-counter*18], width=30, height=13, color=bar_color[column])
        p.text([-15, -15, -15], [37-counter*18], text=[column], text_font_size="9pt", 
               text_align="left", text_baseline="middle")
        counter += 1
    
    #draw the rings and corresponding labels
    labels = np.array(range((label_max+10)/10))*10
    radii = 90 + labels* (200/float(label_max))
    p.circle(0, 0, radius=radii[:-1], fill_color=None, line_color="#E6E6E6")
    p.text(0, radii, [str(z)+str(unit) for z in labels[:-1]], text_font_size="8pt", text_align="center", 
                      text_baseline="middle")
    
    #draw the spokes separating the big wedges
    p.annular_wedge(0, 0, inner_radius-10, outer_radius+10, -big_angle+angles, 
                    -big_angle+angles, color="black")
    
    #draw the labels for the big wedges and angle them correctly
    xr = radii[-1]*np.cos(np.array(-big_angle/2 + angles))
    yr = radii[-1]*np.sin(np.array(-big_angle/2 + angles))   
    label_angle=np.array(-big_angle/2+angles)
    label_angle[label_angle < -np.pi/2] += np.pi   
    p.text(xr, yr, pandas_df[index], angle=label_angle, text_font_size="9pt", 
           text_align="center", text_baseline="middle")

    output_file("example.html", title="example.py")
    show(p)
    
#Test datasets
"""
test1_df = pd.DataFrame(np.random.randn(200,5), columns=list('ABCDE'))
test2_df = pd.DataFrame(np.random.uniform(0,10,200), columns=list('A'))
X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=0)
test3_df = pd.DataFrame(X)
test4_df = pd.DataFrame({'Time': ['0 days 10:22:00.000000000', '1 days 00:00:00.002700000', '3 days 00:15:10.00550000', '0 days 00:00:10.200000000']})
survivor = pd.read_csv("C:\Users\Zachery McKinnon\Documents\survivor_demographics.csv")

print PCA_and_kmeans(test3_df, 3)
print time_split(test4_df, 'Time')
print histogram(test2_df, 'A', 10)
print histogram(test1_df, 'B', 25)
"""
