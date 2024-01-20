from python_packages import *

def list_all_files(root): 
    files = []
    pattern = '*.ct'

    for path, subdirs, files  in os.walk(root):
        for file in files: 
            if fnmatch(file, pattern):
                files.append(os.path.join(path, file))
    return files





#NOTE - Below are only tested with RNAStralign data set. Change/check if other are added

def singleFamily(family: str, file_list: list) -> list:
    '''
    Returns list of files from a single specified family
    '''
    files = []
    for file in file_list: 
        if family in file:
            files.append(file) 

    return files

def leaveOneFamilyOut(family: str, file_list: list):
    '''
    Returns list of files from all families except the specified family
    '''
    files = []
    for file in file_list:
        if family not in file:
            files.append(file)
    return files

def pickFromFamilies(data_size: int, file_list: str):
    '''
    Returns a list of files with data_size from EACH family.
    If a family does not have enough data, all data from that family is added.
    '''
    families = list(set([file.split(os.sep)[4] for file in file_list])) #Find all family names and remove duplicates

    data = [[file for file in file_list if family in file] for family in families] #Create list of lists, where each list contains all files from a family

    files = []
    for family in data: #Pick data_size files from each family
      try:
          files+=rd.sample(family, data_size)
      except:
          print("Not enough data in family, adding all")
          files+=family
    return files

def getLength(ct_file):
    '''
    Opens a ct_file and returns the length of the sequence as it appears in the first line.
    '''
    with open(ct_file, "r") as f:
      for line in f:
        length = int(line.split()[0])
        break
    return length

def underGivenLength(length: int, data_size: int, file_list: list):
  '''
  Returns a list of files under length with data_size from EACH family.
  If a family does not have enough data, all data from that family is added.
  '''
  families = list(set([file.split(os.sep)[4] for file in file_list])) #Find all family names and remove duplicates

  data = [[line for line in files if family in line and getLength(line)<length] for family in families] #Create list of lists, where each list contains all files from a family

  files = []
  for family in data: #Pick data_size files from each family
      try:
          files+=rd.sample(family, data_size)
      except:
          print("Not enough data in family, adding all")
          files+=family
  return files
