# CLUSTERING.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Fits k-means and k-nearest neighbors (knn) clusters, both from scratch and using the built-in functionality in sklearn
# Training data: synthetic and randomly generated points in the shape of a snowman (three stacked circles of different radii)

import time
import math
import numpy as np
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.backends.backend_pdf import PdfPages

time0 = time.time()
np.random.seed(1776)
ver = ''  # version (empty or integer)

topic = 'Clustering'
topic_underscore = topic.replace(' ','_')

#--------------#
#  PARAMETERS  #
#--------------#

kmeans_clusters = 3        # number of k-means clusters to fit
knn_clusters = 5           # number of knn clusters to fit
iterations = 200           # sets of random initial centroids (k-means)
steps_per_iteration = 20   # steps to go on each iteration (k-means)
granularity = 0.1          # spatial granularity of sample space (knn)
x_max = 10                 # x upper bound (region begins at origin)
y_max = 10                 # y upper bound (region begins at origin)
density = 50               # spatial density of training points (per 1 x 1 area)

decimal_places = 3

#-------------#
#  FUNCTIONS  #
#-------------#

def format_plot():
    plt.legend(loc='upper right', fontsize=legend_size, facecolor='white', framealpha=1)
    plt.gca().set_aspect('equal')
    plt.xlabel('X', fontsize=axis_labels_size)
    plt.ylabel('Y', fontsize=axis_labels_size)
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.grid(True, alpha=0.5, zorder=0)
    plt.show(True)

def console_print(subtitle, df, first_line, title_top):
    if first_line == 1:
        print(Fore.GREEN + '\033[1m' + '\n' + title_top + Style.RESET_ALL)
    print(Fore.GREEN + '\033[1m' + '\n' + subtitle + Style.RESET_ALL)
    print(df)
    
def txt_export(subtitle, df, f, first_line, title_top):
    if first_line == 1:
        print(title_top, file=f)
    print('\n' + subtitle, file=f)
    print(df, file=f)

#-----------------#
#  TRAINING DATA  #
#-----------------#

total_circles = 3  # for the snowman

cluster_list = ['cluster ' + str(c + 1) for c in range(total_circles)]

# circle radii
radius_1 = 2.25
radius_2 = 1.5
radius_3 = 1

# circle centroids (circle areas are stacked on top of each other)
x_circles_center = x_max / 2
y_circle_1_center = radius_1
y_circle_2_center = (2 * radius_1) + radius_2
y_circle_3_center = (2 * radius_1) + (2 * radius_2) + radius_3

# set number of points so spatial density of points remains generally constant throughout
n_pts_circle_1 = int(density * math.pi * radius_1**2)
n_pts_circle_2 = int(density * math.pi * radius_2**2)
n_pts_circle_3 = int(density * math.pi * radius_3**2)

# initialize
circle_1 = pd.DataFrame()
circle_2 = pd.DataFrame()
circle_3 = pd.DataFrame()

# randomly generate points within the bounds of each circle area, in circular coordinates (offset, theta)
circle_1['offset'] = np.random.uniform(low=0, high=radius_1, size=n_pts_circle_1)  # circle centroid is an offset of zero
circle_2['offset'] = np.random.uniform(low=0, high=radius_2, size=n_pts_circle_2)
circle_3['offset'] = np.random.uniform(low=0, high=radius_3, size=n_pts_circle_3)
circle_1['theta'] = np.random.uniform(low=0, high=2*math.pi, size=n_pts_circle_1)  # angle spans entire circle (zero to 2*pi radians)
circle_2['theta'] = np.random.uniform(low=0, high=2*math.pi, size=n_pts_circle_2)
circle_3['theta'] = np.random.uniform(low=0, high=2*math.pi, size=n_pts_circle_3)

# convert points from circular (offset, theta) to cartesian (x, y) coordinates
circle_1['x'] = x_circles_center + circle_1['offset'] * np.cos(circle_1['theta'])
circle_2['x'] = x_circles_center + circle_2['offset'] * np.cos(circle_2['theta'])
circle_3['x'] = x_circles_center + circle_3['offset'] * np.cos(circle_3['theta'])
circle_1['y'] = y_circle_1_center + circle_1['offset'] * np.sin(circle_1['theta'])
circle_2['y'] = y_circle_2_center + circle_2['offset'] * np.sin(circle_2['theta'])
circle_3['y'] = y_circle_3_center + circle_3['offset'] * np.sin(circle_3['theta'])

# true (actual) group labels
circle_1['actual group'] = 1
circle_2['actual group'] = 2
circle_3['actual group'] = 3

training = pd.concat([circle_1, circle_2, circle_3], axis=0, ignore_index=True)
total_pts_training = len(training)

#--------------------------#
#  K-MEANS - FROM SCRATCH  #
#--------------------------#

wcss_best = math.inf  # within cluster sum of squares (initialize to be worst possible value - positive infinity - because LOWER values are sought)

for i in range(iterations):
    # step 1 - randomly select points in the sample space to use as initial cluster centroids
    x_initial = list(np.random.uniform(low=0, high=x_max, size=kmeans_clusters))
    y_initial = list(np.random.uniform(low=0, high=y_max, size=kmeans_clusters))
    y_initial.sort()
    x_centroids = x_initial.copy()
    y_centroids = y_initial.copy()
    
    # optimize cluster locations for current set of initial centroids
    for t in range(steps_per_iteration):
        distances = pd.DataFrame()
        
        # step 2 - assign each point to cluster with nearest centroid
        for c in range(kmeans_clusters):
            distances['dist cluster ' + str(c + 1)] = np.sqrt((training['x'] - x_centroids[c])**2 + (training['y'] - y_centroids[c])**2)  # distance to cluster centroid
        distances['min dist'] = distances.min(axis=1)  # minimum distance (closest cluster)
        distances['dist sqrd'] = distances['min dist']**2
        for c in range(kmeans_clusters):
            distances.loc[distances['dist cluster ' + str(c + 1)] == distances['min dist'], 'pred cluster'] = c + 1  # best cluster for each point (minimum distance)
        clusters = pd.concat([training, distances], axis=1)
        
        # step 3 - average coordinates in each cluster become the new cluster centroids
        for c in range(kmeans_clusters):
            if len(clusters.loc[clusters['pred cluster'] == c + 1]) > 0:  # leave coordinates unchanged if no points were assigned to cluster
                x_centroids[c] = stat.mean(clusters.loc[clusters['pred cluster'] == c + 1, 'x'])
                y_centroids[c] = stat.mean(clusters.loc[clusters['pred cluster'] == c + 1, 'y'])
        
        wcss = sum(clusters['dist sqrd'])  # within cluster sum of squares (variance in position about cluster centroids)
        
        # compare current iteration to running global best solution
        if wcss < wcss_best:
            wcss_best = wcss
            x_centroids_kmeans = x_centroids.copy()
            y_centroids_kmeans = y_centroids.copy()
            kmeans_fit = clusters.copy(deep=True)
del clusters, wcss, x_centroids, y_centroids

#----------------------#
#  KNN - FROM SCRATCH  #
#----------------------#

# initalize
knn_full = pd.DataFrame()
knn_fit = pd.DataFrame(columns=['x', 'y', 'pred cluster'])
boundary_c1_c2 = pd.DataFrame(columns=['x', 'y'])
boundary_c2_c3 = pd.DataFrame(columns=['x', 'y'])
prior_group = 0
prior_x = 0
prior_y = 0

# knn cluster regions (by plurality vote of k-nearest neighbors) (for all points in the sample space)
for x_pt in np.arange(0, x_max + granularity, granularity):
    for y_pt in np.arange(0, y_max + granularity, granularity):
        # distance to each point (ascending)
        knn_full['distance'] = np.sqrt((training['x'] - x_pt)**2 + (training['y'] - y_pt)**2)  # distance to each point (Pythagorean)
        knn_full.sort_values(by=['distance'], ignore_index=False, inplace=True)  # sort so nearest neighbors are at top
        knn_full['actual group'] = training['actual group']
        knn_subset = knn_full.head(knn_clusters)  # keep only k-nearest neighbor points
        knn_group = knn_subset['actual group'].mode()[0]  # mode represents the plurality vote
        knn_fit.loc[len(knn_fit)] = [x_pt, y_pt, knn_group]
        
        # assess cluster region boundary (if cluster assignment changed relative to past iteration) (sets cluster boundary to midpoint)
        if (knn_group != prior_group) and (y_pt != 0):
            if (prior_group == 1) and (knn_group == 2):
                boundary_c1_c2.loc[len(boundary_c1_c2)] = [0.5 * (x_pt + prior_x), 0.5 * (y_pt + prior_y)] 
            elif (prior_group == 2) and (knn_group == 3):
                boundary_c2_c3.loc[len(boundary_c2_c3)] = [0.5 * (x_pt + prior_x), 0.5 * (y_pt + prior_y)] 
        
        # save current
        prior_x = x_pt
        prior_y = y_pt
        prior_group = knn_group
del prior_x, prior_y, prior_group, x_pt, y_pt, knn_full, knn_subset, knn_group

total_pts_sample_space = len(knn_fit)

# sample space bounds
boundary_c1_c2['y min'] = 0
boundary_c2_c3['y min'] = 0
boundary_c1_c2['y max'] = y_max
boundary_c2_c3['y max'] = y_max

#--------------------#
#  SKLEARN CLUSTERS  #
#--------------------#

# k-means predictions
kmeans_model_alt = KMeans(n_clusters=kmeans_clusters, n_init='auto').fit(training[['x', 'y']])
kmeans_fit_alt = pd.DataFrame(kmeans_model_alt.labels_, columns=['pred cluster'])
kmeans_fit_alt['pred cluster'] += 1
kmeans_fit_alt[['x', 'y']] = training[['x', 'y']]

# k-means centroids
kmeans_centroids_alt = pd.DataFrame(kmeans_model_alt.cluster_centers_, columns=['x sklearn', 'y sklearn'])
kmeans_centroids_alt.sort_values(by=['y sklearn'], ignore_index=True, inplace=True)
kmeans_centroids_alt = round(kmeans_centroids_alt, decimal_places)
kmeans_centroids_alt.index = cluster_list

# knn test data (entire sample space)
knn_sample_space = pd.DataFrame(columns=['x', 'y'])
for x_pt in np.arange(0, x_max + granularity, granularity):
    for y_pt in np.arange(0, y_max + granularity, granularity):
        knn_sample_space.loc[len(knn_sample_space)] = [x_pt, y_pt]
del x_pt, y_pt

# knn predictions
knn_fit_alt = pd.DataFrame()
knn_fit_alt[['x', 'y']] = knn_sample_space
knn_model_alt = KNeighborsClassifier(n_neighbors=knn_clusters).fit(training[['x', 'y']], training['actual group'])
knn_fit_alt['pred cluster'] = knn_model_alt.predict(knn_fit_alt[['x', 'y']])

#-------------------#
#  SUMMARY RESULTS  #
#-------------------#

# k-means accuracy (from scratch)
kmeans_summary = pd.DataFrame()
kmeans_fit['correct'] = [int(kmeans_fit['pred cluster'][i] == kmeans_fit['actual group'][i]) for i in kmeans_fit.index]
kmeans_correct = kmeans_fit.loc[kmeans_fit['correct'] == 1]
kmeans_misclassified = kmeans_fit.loc[kmeans_fit['correct'] == 0]
kmeans_accuracy = round(100 * len(kmeans_correct) / total_pts_training, 2)
kmeans_summary.index = ['accuracy (%)', 'total points', 'correct (#)', 'misclassified (#)']
kmeans_summary['from scratch'] = [str(kmeans_accuracy), total_pts_training, len(kmeans_correct), len(kmeans_misclassified)]

# k-means centroids (from scratch)
kmeans_centroids = pd.DataFrame()
kmeans_centroids['x from scratch'] = x_centroids_kmeans
kmeans_centroids['y from scratch'] = y_centroids_kmeans
kmeans_centroids.sort_values(by=['y from scratch'], ignore_index=True, inplace=True)
kmeans_centroids = round(kmeans_centroids, decimal_places)
kmeans_centroids.index = cluster_list

# k-means centroids (merged)
kmeans_centroids_merged = pd.concat([kmeans_centroids, kmeans_centroids_alt], axis=1)
kmeans_centroids_merged = kmeans_centroids_merged[['x from scratch', 'x sklearn', 'y from scratch', 'y sklearn']]

# knn comparison (from scratch, sklearn)
knn_summary = pd.DataFrame()
for df, suffix in zip([knn_fit, knn_fit_alt], ['from scratch', 'sklearn']):
    df2 = pd.DataFrame(df['pred cluster'])
    df2['cnt'] = 1
    df3 = df2.groupby('pred cluster').count()
    knn_summary['% ' + suffix] = round(100 * df3['cnt'] / len(df), 2)
del df, df2, df3
knn_summary.index = cluster_list

#---------#
#  PLOTS  #
#---------#

# parameters
title_size = 11
axis_labels_size = 8
legend_size = 8
point_size = 12
big_point_size = 40
alpha_val = 0.5

# training data
fig1 = plt.figure()
plt.title(topic + ' - training data (snowman)', fontsize=title_size, fontweight='bold')
plt.scatter(circle_1['x'], circle_1['y'], marker='*', s=point_size, alpha=alpha_val, color='#1f77b4', label='group 1', zorder=5)
plt.scatter(circle_2['x'], circle_2['y'], marker='*', s=point_size, alpha=alpha_val, color='#ff7f0e', label='group 2', zorder=10)
plt.scatter(circle_3['x'], circle_3['y'], marker='*', s=point_size, alpha=alpha_val, color='#2ca02c', label='group 3', zorder=15)
format_plot()

# k-means clusters
fig2 = plt.figure()
plt.title('k-means clusters (k = ' + str(kmeans_clusters) + ')', fontsize=title_size, fontweight='bold')
for c in range(kmeans_clusters):
    x_col = kmeans_fit.loc[kmeans_fit['pred cluster'] == c + 1, 'x']
    y_col = kmeans_fit.loc[kmeans_fit['pred cluster'] == c + 1, 'y']
    plt.scatter(x_col, y_col, marker='*', s=point_size, alpha=alpha_val, label='cluster ' + str(c + 1), zorder=5*(c + 1))
plt.scatter(x_centroids_kmeans, y_centroids_kmeans, marker='*', s=big_point_size, color='black', label='centroid', zorder=20)
format_plot()

# k-means accuracy
fig3 = plt.figure()
plt.title('k-means clusters accuracy (k = ' + str(kmeans_clusters) + ')', fontsize=title_size, fontweight='bold')
plt.scatter(kmeans_correct['x'], kmeans_correct['y'], marker='*', s=point_size, alpha=alpha_val, label='correct', zorder=5)
plt.scatter(kmeans_misclassified['x'], kmeans_misclassified['y'], marker='*', s=point_size, alpha=alpha_val, label='incorrect', zorder=10)
plt.scatter(x_centroids_kmeans, y_centroids_kmeans, marker='*', s=big_point_size, color='black', label='centroid', zorder=15)
format_plot()

# knn clusters
fig4, ax = plt.subplots()
plt.title('KNN cluster regions (k = ' + str(knn_clusters) + ')', fontsize=title_size, fontweight='bold')
ax.fill_between(boundary_c1_c2['x'], boundary_c1_c2['y'], boundary_c1_c2['y min'], alpha=alpha_val, label='cluster 1')
ax.fill_between(boundary_c1_c2['x'], boundary_c2_c3['y'], boundary_c1_c2['y'], alpha=alpha_val, label='cluster 2')
ax.fill_between(boundary_c1_c2['x'], boundary_c2_c3['y max'], boundary_c2_c3['y'], alpha=alpha_val, label='cluster 3')
plt.scatter(training['x'], training['y'], marker='*', s=point_size, alpha=0.4, color='black', label='training')
format_plot()

#----------#
#  EXPORT  #
#----------#

title_top = topic.upper() + ' SUMMARY'
dfs = [kmeans_summary, kmeans_centroids_merged, knn_summary]
df_labels = ['k-means accuracy', 'k-means centroids', 'knn distribution']

# export summary (console, txt)
with open(topic_underscore + '_summary' + ver + '.txt', 'w') as f:
    for out in ['console', 'txt']:
        first_line = 1
        for d in range(len(dfs)):
            subtitle = df_labels[d].upper()
            if out == 'console':
                console_print(subtitle, dfs[d], first_line, title_top)  # console print
            elif out == 'txt':
                txt_export(subtitle, dfs[d], f, first_line, title_top)  # txt file
            first_line = 0
del f, title_top, subtitle

# export plots (pdf)
pdf = PdfPages(topic + '_plots' + ver + '.pdf')
for f in [fig1, fig2, fig3, fig4]:
    pdf.savefig(f)
pdf.close()
del pdf, f

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0


