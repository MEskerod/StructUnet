import os, pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import colorcet as cet

import numpy as np
import pandas as pd


def plot_families(file_dict: dict, output_file = None) -> None:
  """
  Plots the distribution of data (families) based on the given file dictionary.

  This function takes a dictionary `file_dict` where the keys represent data set names and the values are lists of files.
  It plots the distribution of data (families) based on the files in the dictionary.
  The resulting plot shows the density of each family across different data sets.

  If `output_file` is provided, the plot is saved as an image file at the specified path.

  Parameters:
  - file_dict (dict): A dictionary containing file lists for each data set.
  - output_file (str, optional): The path to save the plot as an image file. Defaults to None.

  Returns:
  - None
  """
  file_dict = {key:item for key, item in file_dict.items() if len(item)>0}

  file_lists = [item for _, item in file_dict.items()]
  data_set = [key for key, _ in file_dict.items()]

  # Make a dictionary of RNA families and their distribution across data sets
  RNA_families = {}
  for index, file_list in enumerate(file_lists):
    for file in file_list:
      family = pickle.load(open(file, 'rb')).family
      if family not in RNA_families:
        RNA_families[family] = [0]*len(file_lists)
      
      RNA_families[family][index] += 1

  #Sorts the families based on the total number of sequences across all data sets
  RNA_families = sorted({family:item for family, item in RNA_families.items() if item != ([0]*len(file_lists))}.items(), key=lambda x:x[1])
  values = [[f[1][i]/len(file_lists[i]) for i in range(len(file_lists))] for f in RNA_families]
  families = [" ".join(f[0].split('_')) for f in RNA_families]

  x = np.arange(len(families))
  width = 1/len(file_dict)*0.9

  fig, ax = plt.subplots(figsize=(10,6))
  for index in range(len(file_lists)):
    offset = width*index
    ax.bar(x+offset, [v[index] for v in values], width=width, edgecolor = 'black', label = data_set[index])

  ax.grid(axis = 'y', linestyle = '--')
  ax.set_axisbelow(True)
  ax.set_xticks(x + width / 2)
  ax.set_xticklabels(families, rotation = 45, ha='right')
  if len(file_dict) > 1:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.94))

  ax.set_ylabel("Density")
  ax.set_title('Distribution of data (families)')

  plt.tight_layout()

  if output_file:
    plt.savefig(output_file, bbox_inches = 'tight')


def plot_len_histogram(file_dict: dict, output_file = None) -> None:
  """
  Takes a dictionary of file lists and produces a histogram of the distribution of lengths of sequences.

  Parameters:
  - file_dict (dict): A dictionary where the keys are data set names and the values are lists of file paths.
  - output_file (str, optional): The file path to save the histogram plot. If not provided, the plot will be displayed.

  Returns:
  - None

  Example usage:
  >>> file_dict = {'Data Set 1': ['file1.pkl', 'file2.pkl'], 'Data Set 2': ['file3.pkl']}
  >>> plot_len_histogram(file_dict, output_file='histogram.png')
  """

  file_dict = {key:item for key, item in file_dict.items() if len(item)>0}

  file_lists = [item for _, item in file_dict.items()]
  data_set = [key for key, _ in file_dict.items()]

  # Takes a list of .pkl files and produces a histogram of the distribution of lengths of sequences
  lengths = []
  #Get the lengths of the sequences for each data set
  for file_list in file_lists:
    l = []
    for file in file_list:
      l.append(pickle.load(open(file, 'rb')).length)
    lengths.append(l)

  plt.figure(figsize=(10,6))

  plt.hist(lengths, bins=50, edgecolor='black', label = data_set, zorder = 3, density = True)

  plt.title('Distribution of data (lengths)')
  plt.xlabel('Sequence Length')
  plt.ylabel("Density")
  if len(file_dict) > 1:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.94))
  plt.grid(axis = 'y', linestyle='--', zorder=0)
  plt.tight_layout()

  if output_file:
    plt.savefig(output_file, bbox_inches = 'tight', dpi = 300)

  plt.show()




def plot_time(time, lengths, outputfile = None):
    """
    Plots the time it takes to predict the structure of a sequence using the Nussinov algorithm

    Parameters:
    - time (list): A list of floats representing the time it took to predict the structure of each sequence.
    - lengths (list): A list of integers representing the length of each sequence.
    """
    col = mpl.colormaps['cet_glasbey_dark'].colors[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, time, facecolor='none', edgecolor = 'C0', s=20, linewidths = 1)
    plt.plot(lengths, time, linestyle = '--', linewidth = 0.8)
    plt.xlabel('Sequence length')
    plt.ylabel('Time (s)')
    plt.grid(linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout
    plt.savefig(outputfile, dpi=300, bbox_inches='tight')



def plot_timedict(timedict: dict, lengths: list, outputfile = None) -> None:
    """
    Plots the time it takes to predict the structure of a sequence using different methods (given be the dictionary)
    Makes a scatter plot connected with lines with the time it takes to predict the structure of a sequence using different methods 

    Parameters:
    - timedict (dict): A dictionary where the keys are the names of the methods and the values are lists of floats representing the time it took to predict the structure of each sequence.
    - lengths (list): A list of integers representing the length of each sequence.
    - outputfile (str, optional): The file path to save the plot. If not provided, the plot will be displayed.

    Returns:
    - None
    """

    colors = mpl.colormaps['cet_glasbey_dark'].colors
    
    
    fig, ax = plt.subplots(figsize = (10, 6)) 
    handles = []

    for i, func in enumerate(timedict):
        ax.scatter(x= lengths, y=timedict[func], facecolor='none', edgecolor = colors[i], s=15, linewidths = 1)
        ax.plot(lengths, timedict[func], color = colors[i], linestyle = '--', linewidth = 0.8)
        #Add handles to make a legend
        handles.append(Line2D([0], [0], color = colors[i], linestyle = '--', linewidth = 0.8, marker = 'o', markerfacecolor = 'none', markeredgecolor = colors[i], label = func))

    
    ax.legend(handles = handles, loc = 'upper left', frameon = False)
    ax.grid(linestyle = '--')
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Time (s)")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.tight_layout()
    
    if outputfile: 
        plt.savefig(outputfile, dpi = 300)




def adjacent_values(min: float, max: float, q1: float, q3: float) -> tuple:
  """
  Calculate the lower and upper adjacent values for a given range.

  Parameters:
  - min (float): The minimum value of the range.
  - max (float): The maximum value of the range.
  - q1 (float): The first quartile value.
  - q3 (float): The third quartile value.

  Returns:
  - tuple: A tuple containing the lower and upper adjacent values.

  """
  iqr = q3 - q1
  upper_adjacent_value = np.clip(q3 + 1.5 * iqr, q3, max)
  lower_adjacent_value = np.clip(q1 - 1.5 * iqr, min, q1)
  return lower_adjacent_value, upper_adjacent_value

def find_outliers(df: pd.DataFrame, lower_bounds: list, upper_bounds:list) -> list:
  """
  Finds outliers in a DataFrame based on lower and upper bounds for each column.

  Parameters:
  - df (pd.DataFrame): The DataFrame to search for outliers.
  - lower_bounds (list): A list of lower bounds for each column.
  - upper_bounds (list): A list of upper bounds for each column.

  Returns:
  - list: A list of lists containing the outliers for each column.
  """
  filtered_data = []
  for i, column in enumerate(df.columns):
    filtered_data.append(list(df[column][(df[column] < lower_bounds[i]) | (df[column] > upper_bounds[i])]))
  return filtered_data

def set_axis_style(ax, labels: list, ylabel: str) -> None:
  """
  Set the style of the axis in a matplotlib plot.

  Parameters:
  - ax (matplotlib.axes.Axes): The axes object to modify.
  - labels (list): The labels for the y-axis ticks.
  - ylabel (str): The label for the y-axis.

  Returns:
  None
  """
  ax.set_yticks(np.arange(1, len(labels) + 1), labels=labels, fontsize = 12)
  ax.tick_params(axis='x', labelsize = 12)
  ax.set_ylim(0.25, len(labels) + 0.75)
  ax.set_ylabel(ylabel, fontsize = 10)
  ax.set_xlabel('F1 score', fontsize = 10)
  ax.tick_params(axis= 'both', which = 'both', bottom = False, left = False)
  ax.grid(axis = 'x', linestyle = '--', alpha = 0.3, zorder = 0)
  ax.set_axisbelow(True)

def violin_plot(df: pd.DataFrame, ylabel: str, cmap='cet_glasbey_dark', outputfile=None) -> None:
  """
  Generate a violin plot for the given DataFrame.

  Parameters:
  - df (pd.DataFrame): The DataFrame containing the data to be plotted.
  - ylabel (str): The label for the y-axis.
  - cmap (str, optional): The colormap to be used for coloring the violins. Defaults to 'Accent'.
  - outputfile (str, optional): The file path to save the plot. If not provided, the plot will be displayed.

  Returns:
  None
  """
  #Reverse order of columns such that the first column is the top of the plot
  df = df[df.columns[::-1]]
  
  cmap = mpl.colormaps[cmap].colors

  # Get number of categories to define the size of the plot
  categories = len(df.columns)
  fig, ax = plt.subplots(figsize = (10, categories))

  parts = ax.violinplot(df, showmeans=False, showmedians = False, showextrema = False, vert = False)

  #Change apperance of body
  for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(cmap[i])
    pc.set_edgecolor('black')
    pc.set_alpha(1)
    pc.set_zorder(2)

  #Calculate quantiles
  quartile1, medians, quartile3 = np.percentile(df.values, [25, 50, 75], axis=0)
  df_min, df_max = df.min().min(), df.max().max()
  whiskers = np.array([adjacent_values(df_min, df_max, q1, q3) for q1, q3 in zip(quartile1, quartile3)])
  whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

  outliers = find_outliers(df, whiskers_min, whiskers_max)

  #Add median
  inds = np.arange(1, len(medians) + 1)
  ax.scatter(medians, inds, marker='|', color='white', s=30, zorder=3)

  #Add value as text for mean
  for i, mean_val in enumerate(df.mean()): 
    ax.text(1.005, inds[i], f"{mean_val:.3f}", ha='left', fontsize = 12)

  #Add boxes
  ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5, zorder = 2.5)
  #Add whiskers
  ax.hlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1, zorder = 2.5)

  #Add outliers
  for i in inds:
    ax.scatter(outliers[i-1], np.ones_like(outliers[i-1])*i, marker = 'o', s=8, color = 'black', zorder = 2.5)

  #Set style for the axes
  labels = [name.split('_')[0] for name in df.columns]
  set_axis_style(ax, labels, ylabel) 
  plt.box(False)

  plt.subplots_adjust(bottom=0.15, wspace=0.05)

  plt.tight_layout()

  if outputfile: 
    plt.savefig(outputfile, dpi = 300, bbox_inches = 'tight')
  plt.show()
     