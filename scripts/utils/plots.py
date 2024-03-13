import os, pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_families(file_dict: dict, output_file = None):
  file_dict = {key:item for key, item in file_dict.items() if len(item)>0}

  file_lists = [item for _, item in file_dict.items()]
  data_set = [key for key, _ in file_dict.items()]

  RNA_families = {}

  for index, file_list in enumerate(file_lists):
    for file in file_list:
      family = pickle.load(open(file, 'rb')).family
      if family not in RNA_families:
        RNA_families[family] = [0]*len(file_lists)
      
      RNA_families[family][index] += 1

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
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.94))

  ax.set_ylabel("Density")
  ax.set_title('Distribution of data (families)')

  plt.tight_layout()

  if output_file:
    plt.savefig(output_file, bbox_inches = 'tight')


def plot_len_histogram(file_dict: dict, output_file = None):
  file_dict = {key:item for key, item in file_dict.items() if len(item)>0}

  file_lists = [item for key, item in file_dict.items()]
  data_set = [key for key, item in file_dict.items()]

  # Takes a list of .pkl files and produces a histogram of the distribution of lengths of sequences
  lengths = []

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
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.94))
  plt.grid(axis = 'y', linestyle='--', zorder=0)
  plt.tight_layout()

  if output_file:
    plt.savefig(output_file, bbox_inches = 'tight', dpi = 300)

  plt.show()







def adjacent_values(min, max, q1, q3):
  iqr = q3-q1
  upper_adjacent_value = np.clip(q3 + 1.5 * iqr, q3, max)
  lower_adjacent_value = np.clip(q1 - 1.5 * iqr, min, q1)
  return lower_adjacent_value, upper_adjacent_value

def find_outliers(df, lower_bounds, upper_bounds):
  filtered_data = []
  for i, column in enumerate(df.columns):
    filtered_data.append(list(df[column][(df[column] < lower_bounds[i]) | (df[column] > upper_bounds[i])]))
  return filtered_data

def set_axis_style(ax, labels, ylabel):
    ax.set_yticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_ylim(0.25, len(labels) + 0.75)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis= 'both', which = 'both', bottom = False, left = False)
    ax.grid(axis = 'x', linestyle = '--', alpha = 0.3, zorder = 0)
    ax.set_axisbelow(True)
    plt.box(False)

def violin_plot(df, ylabel, cmap = 'Accent', outputfile = None):
  cmap = mpl.colormaps[cmap].colors

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

  plt.subplots_adjust(bottom=0.15, wspace=0.05)

  plt.tight_layout()

  if outputfile: 
    plt.savefig(outputfile, dpi = 300, bbox_inches = 'tight')
  
  plt.show()
     