import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from pointpats import PointPattern
import glob
import os
import copy
from scipy.spatial import ConvexHull
import sys

#requires scikit-learn==1.0.2

scan=sys.argv[1] # "/full/path/to/scan/data"    
nBins=int(sys.argv[2]) # number of clusters - if this is too big the program will fail - select a smaller number. 


"""
INPUT:
Reqired files in the scan directory:
    
Diffraction and Roughness file.
"*-Diffraction Peaks and Roughness.csv" - this file is exported from PIXLISE and contains the following columns
ID, PMC Energy (keV),   Energy Start (keV), Energy End (keV),   Effect Size,    Baseline Variation, Global Difference,  Difference Sigma,   Peak Height,    Detector,   Channel,    Status

Beam Locations file
This file is exported from PIXLISE and ends with "*-beam-locations.csv" and contains the x,y postions in um of each PMC in the first three columns: "PMC, x, y"

OUTPUT:
"scan/PIXLISE_PMC_List.txt" : a single string list of PMCs in spatially coherent regions to copy/paste into PIXLISE for further analysis
"scan/SpatiallyCoherentPMCs.txt" : a 3 colums csv with PMC, x (mm), y (mm) for all PMCs in the spatially coherent regions
"scan/SpatiallyCoherentPMCs.png" : a basic plot of the PMCs in spatially coherent regions together with the scan footprint


WARNINGS:
    
If "UnboundLocalError: local variable 'number_of_bins_output' referenced before assignment" occurs, select fewer initial bins (sys.argv[3])

"""

def find_energy(theta_width, d):
    """
    Find the energy given a diffraction angle and lattice d-spacing using the energy-dispersive Bragg Equation 
    (Equation 3 in Dragoi & Dragoi 2018)

    theta_width - diffraction angle in radians ('float')
    d - lattice d-spacing in angstroms (numpy.float64)
    """

    return 12.398/(np.sin(theta_width)*2*d)

def find_energy_width(energy):
    """
    Use PIXL's geometry and its convergence angle (8 degrees) to find the energy width for each energy

    energy - diffracting X-ray energy in keV (numpy.float64)
    """

    theta_width = 8/2*np.pi/180
    theta = 160/2*np.pi/180
    theta_width_min = theta-theta_width
    theta_width_max = theta+theta_width

    d_theta = 12.398/(2*energy*np.sin(theta))

    energy_theta_width_min = find_energy(theta_width_min, d_theta)
    energy_theta_width_max = find_energy(theta_width_max, d_theta)

    return energy_theta_width_min-energy_theta_width_max

def create_clusters(peaks, detector_letter, min_number_of_bins):
    """
    From a list of energy-dispersive diffraction peaks create monocrystalline regions.
    This is performed by finding the maximum number of energy bins centred around maximum diffraction peak heights and returning the number of partitions and corresponding xy coordinates, energies, peak heights and pmcs.
    Orenstein, B.J., Jones, M.W.M., Flannery, D.T., Wright, A.P., Davidoff, S., Tice, M.M., Nothdurft, L., Allwood, A.C., 2024. In-situ mapping of monocrystalline regions on Mars. Icarus 420, 116202. https://doi.org/10.1016/j.icarus.2024.116202

    peaks - Dataframe of diffraction peak information (pandas.core.frame.DataFrame)
    detector letter - Spectrometer detector 'A' or 'B' (str)
    min_number_of_bins - The minimum number of bins used for partitioning (int)    
    """

    try:

        for number_of_bins in np.arange(min_number_of_bins, 1000):
            
            print(number_of_bins)
        
            peaks = peaks[peaks.Detector==detector_letter]
        
            diffraction_energies = peaks['Energy (keV)'].values
            peak_height = peaks['Peak Height'].values
        
            diffraction_indices =  peaks['ID'].values
        
            diffraction_energy_arr = [np.nan]*(number_of_bins+1)     #Array of diffraction energies for each bin
            diffraction_energy_arr[0] = diffraction_energies #Start the first diffraction energy bin off with all of the diffraction energies
        
            peak_height_bin = [np.nan]*(number_of_bins+1) #Array of peak heights for each bin
            peak_height_bin[0] = peak_height      #Start the first peak height bin with all of the peak heighs 
        
            diffraction_indices_arr = [np.nan]*(number_of_bins+1)    
            diffraction_indices_arr[0] = diffraction_indices 
        
            clusters_x_arr = []
            clusters_y_arr = []
            clusters_energies_arr = []
            clusters_peak_height_arr = []
            clusters_pmcs_arr = []
        
            for i in range(number_of_bins):   #For all bins
        
                clusters_x = []
                clusters_y = []
                clusters_energies = []
                clusters_peak_height = []
                clusters_pmcs = []
        
                next_bin_energy = []     #Used to contain the energies available for the next bin
                next_bin_peak_height = [] #Used to contain the energies available for the next bin
                next_bin_diffraction_indices = [] #Used to contain the diffraction indices available for the next bin
                
                center_bin_index = np.nanargmax(peak_height_bin[i]) #The index of the maximum value of the peak heights
                center_bin_energy = diffraction_energy_arr[i][center_bin_index] #The energy corresponding to this index
        
                bin_width = find_energy_width(center_bin_energy)
        
                for energy_index in np.arange(len(diffraction_energy_arr[i])): #For all energies
        
                    if diffraction_energy_arr[i][energy_index] >= center_bin_energy-bin_width and diffraction_energy_arr[i][energy_index] <= center_bin_energy+bin_width:
                    #If a diffraction energy value in the current bin is within the bin range
                        next_bin_energy.append(np.nan)     #Set the energy in the next bin to nan
                        next_bin_peak_height.append(np.nan) #Set the peak height in the next bin to nan
                        next_bin_diffraction_indices.append(np.nan) #Set the peak height in the next bin to nan
        
                        clusters_x.append(peaks[peaks['ID']==diffraction_indices_arr[i][energy_index]]['x_coord'].values[0])
                        clusters_y.append(peaks[peaks['ID']==diffraction_indices_arr[i][energy_index]]['y_coord'].values[0])
                        clusters_peak_height.append(peaks[peaks['ID']==diffraction_indices_arr[i][energy_index]]['Peak Height'].values[0])
                        clusters_energies.append(peaks[peaks['ID']==diffraction_indices_arr[i][energy_index]]['Energy (keV)'].values[0])
                        clusters_pmcs.append(peaks[peaks['ID']==diffraction_indices_arr[i][energy_index]]['PMC'].values[0])
        
                    else:
                        next_bin_energy.append(diffraction_energy_arr[i][energy_index]) #Keep the energy in the next bin
                        next_bin_peak_height.append(peak_height_bin[i][energy_index])     #Keep the peak height in the next bin
                        next_bin_diffraction_indices.append(diffraction_indices_arr[i][energy_index])     #Keep the peak height in the next bin
                
                clusters_x_arr.append(np.array(clusters_x)[np.argsort(clusters_peak_height)])
                clusters_y_arr.append(np.array(clusters_y)[np.argsort(clusters_peak_height)])
                clusters_energies_arr.append(np.array(clusters_energies)[np.argsort(clusters_peak_height)])
                clusters_peak_height_arr.append(np.array(clusters_peak_height)[np.argsort(clusters_peak_height)])
                clusters_pmcs_arr.append(np.array(clusters_pmcs)[np.argsort(clusters_peak_height)])
        
                peak_height_bin[i+1]=next_bin_peak_height      #Set the next bin's available peak heights
                diffraction_energy_arr[i+1]=next_bin_energy  #Set the next bin's available diffraction energies
                diffraction_indices_arr[i+1]=next_bin_diffraction_indices  #Set the next bin's available diffraction energies
        
            number_of_bins_output = np.copy(number_of_bins)
            clusters_x_arr_output = copy.copy(clusters_x_arr) #np.copy can't copy an inhomegenous array, need to use copy.copy
            clusters_y_arr_output = copy.copy(clusters_y_arr)
            clusters_energies_arr_output = copy.copy(clusters_energies_arr)
        
            clusters_peak_height_arr_output = copy.copy(clusters_peak_height_arr)
            clusters_pmcs_arr_output = copy.copy(clusters_pmcs_arr)
        
            min_clusters_x = []
            for x in clusters_x_arr:
                min_clusters_x.append(np.min(x))
        
            max_clusters_x = []
            for x in clusters_x_arr:
                max_clusters_x.append(np.max(x))
        
            min_clusters_y = []
            for y in clusters_y_arr:
                min_clusters_y.append(np.min(y))
        
            max_clusters_y = []
            for y in clusters_y_arr:
                max_clusters_y.append(np.max(y))
        
            min_x = np.min(min_clusters_x)
            max_x = np.max(max_clusters_x)
            min_y = np.min(min_clusters_y)
            max_y = np.max(max_clusters_y)

    except:
        return number_of_bins_output, clusters_x_arr_output, clusters_y_arr_output, clusters_energies_arr_output, clusters_peak_height_arr_output, clusters_pmcs_arr_output, min_x, max_x, min_y, max_y, diffraction_energies, peak_height

def create_clusters_wrapper(scan, min_number_of_bins):
    """
    Load and prepare the beam locations and machine learning 
    identified diffraction peaks for clustering. Run clustering.

    scan - PIXL scan name as exported from PIXLISE https://www.pixlise.org/ (str)
    min_number_of_bins - The minimum number of bins used for partitioning (int)
    """
    beam_locations_image_data = glob.glob(os.path.join(scan , '*-beam-locations.csv'))[0]
    beam_locations_file = beam_locations_image_data
    beam_locations = np.genfromtxt(beam_locations_file, delimiter=",", skip_header=1)

    pmcs = beam_locations[:,0]
    x_coord_xy = beam_locations[:,1]*-1*1e3
    y_coord_xy = beam_locations[:,2]*1e3

    x_coord = x_coord_xy
    y_coord = y_coord_xy
    anomolies_file=glob.glob(os.path.join(scan , '*-Diffraction Peaks and Roughness.csv'))[0]
    anomalies = pd.read_csv(anomolies_file)

    peaks = anomalies.loc[anomalies['Status'] == 'Diffraction Peak']

    x_coord_df_column = []
    y_coord_df_column = []

    for pmc_val in peaks['PMC'].values:
        for pmc_index in range(len(pmcs)):
            if int(pmc_val)==int(pmcs[pmc_index]):
                x_coord_df_column.append(x_coord[pmc_index])
                y_coord_df_column.append(y_coord[pmc_index])

    peaks['x_coord'] = x_coord_df_column
    peaks['y_coord'] = y_coord_df_column

    peaks['indices'] = peaks.index.values

    number_of_bins_A, clusters_x_arr_A, clusters_y_arr_A, clusters_energies_arr_A, clusters_peak_height_arr_A, clusters_pmcs_arr_A, min_x_A, max_x_A, min_y_A, max_y_A, diffraction_energies_A, peak_height_A = create_clusters(peaks, 'A', min_number_of_bins)
    number_of_bins_B, clusters_x_arr_B, clusters_y_arr_B, clusters_energies_arr_B, clusters_peak_height_arr_B, clusters_pmcs_arr_B, min_x_B, max_x_B, min_y_B, max_y_B, diffraction_energies_B, peak_height_B = create_clusters(peaks, 'B', min_number_of_bins)

    return number_of_bins_A, clusters_x_arr_A, clusters_y_arr_A, clusters_energies_arr_A, clusters_peak_height_arr_A, clusters_pmcs_arr_A, min_x_A, max_x_A, min_y_A, max_y_A, diffraction_energies_A, peak_height_A, number_of_bins_B, clusters_x_arr_B, clusters_y_arr_B, clusters_energies_arr_B, clusters_peak_height_arr_B, clusters_pmcs_arr_B, min_x_B, max_x_B, min_y_B, max_y_B, diffraction_energies_B, peak_height_B

def cluster_pmcs(data, pmcs, epsilon, N): #DBSCAN, euclidean distance

    """
    Adapted from Stack Overflow
    https://stackoverflow.com/questions/47974874/algorithm-for-grouping-points-in-given-distance
    Question author: Banana https://stackoverflow.com/users/8986622/banana
    Answer author: Banana https://stackoverflow.com/users/8986622/banana

    Cluster coordinates and pmcs using the DBSCAN algorithm into spatially coherent groups by seeing if a certain number of nearest neighbours are within a specified distance

    data - list of coordinates (numpy.ndarray)            
    pmcs - list of pmcs (pandas.core.series.Series)
    epsilon - spatial coherence distance (float)
    N - number of nearest neighbours (int)
    """

    db     = DBSCAN(eps=epsilon, min_samples=N).fit(data)
    labels = db.labels_ #labels of the found clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #number of clusters
    clusters   = [data[labels == i] for i in range(n_clusters)] #list of clusters
    pmcs_clusters   = [pmcs[labels == i] for i in range(n_clusters)] #list of clusters
    return clusters, n_clusters, pmcs_clusters

def spatially_coherent_pmcs(scan, min_number_of_bins):

    """
    Finds spatially coherent monocrystalline regions

    scan - PIXL scan name as exported from PIXLISE https://www.pixlise.org/ (str)
    min_number_of_bins - The minimum number of bins used for partitioning (int)    
    """

    number_of_bins_A, clusters_x_arr_A, clusters_y_arr_A, clusters_energies_arr_A, clusters_peak_height_arr_A, clusters_pmcs_arr_A, min_x_A, max_x_A, min_y_A, max_y_A, diffraction_energies_A, peak_height_A, number_of_bins_B, clusters_x_arr_B, clusters_y_arr_B, clusters_energies_arr_B, clusters_peak_height_arr_B, clusters_pmcs_arr_B, min_x_B, max_x_B, min_y_B, max_y_B, diffraction_energies_B, peak_height_B = create_clusters_wrapper(scan,  min_number_of_bins)

    clusters_peak_height_arr_combine = clusters_peak_height_arr_A + clusters_peak_height_arr_B
    clusters_energies_arr_combine = clusters_energies_arr_A + clusters_energies_arr_B
    clusters_x_arr_combine = clusters_x_arr_A + clusters_x_arr_B
    clusters_y_arr_combine = clusters_y_arr_A + clusters_y_arr_B
    clusters_pmcs_combine = clusters_pmcs_arr_A + clusters_pmcs_arr_B

    #The following lines until line 288 are adapted from Stack Overflow
    #https://stackoverflow.com/questions/21816084/manually-sort-a-list-of-integers
    #Question author: Kordan9090 https://stackoverflow.com/users/3316774/kordan9090
    #Answer author: Tyler Dane https://stackoverflow.com/users/7781935/tyler-dane

    ## Traverse through all list elements
    for i in range(len(clusters_peak_height_arr_combine)):

    #    # Traverse the list from 0 to n-i-1
    #    # (The last element will already be in place after first pass, so no need to re-check)
        for j in range(0, len(clusters_peak_height_arr_combine)-i-1):

            # Swap if current element is greater than next
            if np.sum(clusters_peak_height_arr_combine[j]) < np.sum(clusters_peak_height_arr_combine[j+1]):
                clusters_peak_height_arr_combine[j], clusters_peak_height_arr_combine[j+1] = clusters_peak_height_arr_combine[j+1], clusters_peak_height_arr_combine[j]
                clusters_energies_arr_combine[j], clusters_energies_arr_combine[j+1] = clusters_energies_arr_combine[j+1], clusters_energies_arr_combine[j]
                clusters_x_arr_combine[j], clusters_x_arr_combine[j+1] = clusters_x_arr_combine[j+1], clusters_x_arr_combine[j]
                clusters_y_arr_combine[j], clusters_y_arr_combine[j+1] = clusters_y_arr_combine[j+1], clusters_y_arr_combine[j]
                clusters_pmcs_combine[j], clusters_pmcs_combine[j+1] = clusters_pmcs_combine[j+1], clusters_pmcs_combine[j]

    #UNIQUE SPATIALLY COHERENT REGIONS

    x_vals_arr = []
    y_vals_arr = []
    pmcs_vals_arr = []

    for partition_index in np.arange(len(clusters_x_arr_combine)):

        df_unique_x_y_pmcs = pd.DataFrame({'x': clusters_x_arr_combine[partition_index], 'y': clusters_y_arr_combine[partition_index], 'pmc': clusters_pmcs_combine[partition_index]}).drop_duplicates()

        unique_pmcs = df_unique_x_y_pmcs['pmc']

        coords = np.swapaxes(np.vstack((df_unique_x_y_pmcs['x'], df_unique_x_y_pmcs['y'])), 0, 1)

        coherence_distance = 0.22

        clusters, n_clusters, clusters_pmcs = cluster_pmcs(coords, unique_pmcs, coherence_distance, 3)

        for i in np.arange(len(clusters)):
            cluster_coords = clusters[i]

            x_vals_arr.append(cluster_coords[:,0])
            y_vals_arr.append(cluster_coords[:,1])
            pmcs_vals_arr.append(clusters_pmcs[i])

    return pmcs_vals_arr, x_vals_arr, y_vals_arr

def flatten(arr):
    """
    Flatten nested lists

    arr - nested list (list)
    """

    flat_arr = []
    for vals in arr:
        for val in vals:
            flat_arr.append(val)

    return flat_arr

def erode(pmcs, x, y, erode_distance):
    """
    Erode the edges of a monocrystalline region by a given distance

    pmcs - pmcs (numpy.ndarray)
    x - x coordinates (numpy.ndarray)
    y - y coordinates (numpy.ndarray)
    erode_distance - distance to erode (float)
    """

    coords = np.swapaxes(np.array([x, y]), 0, 1)
    pp = PointPattern(coords)

    x_eroded = np.array(x)[np.where(pp.knn(3)[1][:,2]<erode_distance)]
    y_eroded = np.array(y)[np.where(pp.knn(3)[1][:,2]<erode_distance)]

    pmcs_eroded = pmcs[np.where(pp.knn(3)[1][:,2]<erode_distance)]

    return pmcs_eroded, x_eroded, y_eroded

def get_crystal_pmcs(scan, min_number_of_bins):

    """
    Finds spatially coherent monocrystalline regions, flattens the regions and erodes the regions, returning a list of eroded region pmcs

    scan - PIXL scan name as exported from PIXLISE https://www.pixlise.org/ (str)
    min_number_of_bins - The minimum number of bins used for partitioning (int)
    """

    crystalline_pmcs, x_vals_arr, y_vals_arr = spatially_coherent_pmcs(scan, min_number_of_bins)
    
    df_unique = pd.DataFrame({'pmc': flatten(crystalline_pmcs), 'x': flatten(x_vals_arr), 'y': flatten(y_vals_arr)}).drop_duplicates()

    pmcs_eroded, x_eroded, y_eroded = erode(df_unique['pmc'].to_numpy(), df_unique['x'].to_numpy(), df_unique['y'].to_numpy(), 0.14)

    return pmcs_eroded, x_eroded, y_eroded



[pmcs_eroded, x_eroded, y_eroded] = get_crystal_pmcs(scan,  nBins)

#save the PMC list as a comma seperated list for PIXLISE import
#convert the array to strings
pmc_arrstr = np.char.mod('%i', pmcs_eroded)
pmc_str = ",".join(pmc_arrstr)
pmcListName=os.path.join(scan,'PIXLISE_PMC_List.txt')
text_file = open(pmcListName, "w")
text_file.write(pmc_str)
text_file.close()
#save the PMC, X, Y data for use later if needed
outDataName=os.path.join(scan,'SpatiallyCoherentPMCs.txt')
np.savetxt(outDataName,np.transpose([pmcs_eroded, x_eroded, y_eroded]),delimiter=',',header='PMC, x (mm), y(mm)')

#plot the spatially coherent regions
beam_locations_image_data = glob.glob(os.path.join(scan , '*-beam-locations.csv'))[0]
beam_locations_file = beam_locations_image_data
beam_locations = np.genfromtxt(beam_locations_file, delimiter=",", skip_header=1)
Xvals = beam_locations[:,1]*-1*1e3
Yvals = beam_locations[:,2]*1e3
XY=np.transpose([Xvals,Yvals])
hull = ConvexHull(XY)

for simplex in hull.simplices:
    plt.plot(XY[simplex, 0], XY[simplex, 1], 'k-')
plt.scatter(x_eroded,y_eroded)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.axis('image')
outFigName=os.path.join(scan,'SpatiallyCoherentPMCs.png')
plt.savefig(outFigName)

    
