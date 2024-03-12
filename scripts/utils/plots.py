import os, pickle
import matplotlib.pyplot as plt
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
     