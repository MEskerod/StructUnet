
import os, pickle
from gwf import Workflow, AnonymousTarget

### EXPERIMENTS ###
def make_experiment_data(matrix_type): 
    """
    Make data for experiments with either 8, 9 or 17 channels. 
    Test sets contains a sequences under 500 and no more than 5000 sequences from each family.
    """
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', f'experiment{matrix_type}.tar.gz')]
    options = {"memory":"16gb", "walltime":"03:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/experiment_files.py {matrix_type}
    tar -czf data/experiment{matrix_type}.tar.gz data/experiment{matrix_type}
    rm -r data/experiment{matrix_type}""".format(matrix_type = matrix_type)
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def postprocess_time(): 
    """
    Time postprocessing methods
    """
    inputs = []
    outputs = [os.path.join('results', 'postprocess_time.csv'),
               os.path.join('figures', 'postprocess_time.png')]
    options = {"memory": "16gb", "walltime": "36:00:00", "account":"RNA_Unet", "cores": 4}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/time_postprocessing.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def convert_time():
    """
    Time matrix conversion with 8, 9 and 17 channels    
    """ 
    inputs = []
    outputs = [os.path.join('results', 'convert_time.csv'),
               os.path.join('figures', 'convert_time.png')]
    options = {"memory": "16gb", "walltime": "24:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/time_matrix_conversion.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


### TRAINING ###
def make_complete_set(): 
    """
    Convert all data to matrices and save namedtuple as pickle files
    """
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', 'train.pkl'),
               os.path.join('data', 'valid.pkl'),
               os.path.join('data', 'test.pkl'),
               os.path.join('figures', 'length_distribution.png'),
               os.path.join('figures', 'family_distribution.png')]
    options = {"memory":"16gb", "walltime":"6:00:00", "account":"RNA_Unet", "cores":4}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/complete_dataset.py
    tar -czf data/test_files.tar.gz data/test_files"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)    

def train_model_small(files): 
    """
    Train the model on the entire data set
    """
    inputs = ['data/complete_set.tar.gz']
    outputs = ['RNA_Unet.pth']
    options = {"memory":"8gb", "walltime":"168:00:00", "account":"RNA_Unet", "gres":"gpu:1", "queue":"gpu"} #NOTE - Think about memory and walltime and test GPU
    spec = """CONDA_BASE=$(conda info --base)
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate RNA_Unet

    echo "Job ID: $SLURM_JOB_ID\n"
    nvidia-smi -L
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    nvcc --version
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "Training neural network"
    python3 scripts/training.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def test_model(files): 
    """
    Test the model of the test set and time it
    """
    inputs = ['RNA_Unet.pth'] + files
    outputs = ['results/test_scores.csv'] + [file.replace('data/test_files', 'steps/RNA_Unet') for file in files] #TODO - Add paths for plots and change path to csv
    options = {"memory":"16gb", "walltime":"24:00:00", "account":"RNA_Unet"} #NOTE - Think about memory and walltime
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/predict_test.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

### EVALUATION ###

def evaluate_hotknots():
    """
    Evaluate the hotknots post-processing with different hyper-parameters
    """
    inputs = [os.path.join('data', 'test_RNA_sample', file) for file in os.listdir('data/test_RNA_sample')]
    outputs = ['results/F1_hotknots.csv', 
               'figures/F1_hotknots.png', 
               'figures/time_hotknots.png', 
               'results/time_hotknots.csv']
    options = {"memory":"8gb", "walltime":"48:00:00", "account":"RNA_Unet", "cores":1}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/evaluate_hotknot.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_hotknots(files): 
    """
    Predict structure with hotknots 
    """
    inputs = [files]
    outputs = [os.path.join('steps', 'hotknots', os.path.basename(file)) for file in files]
    options = {"memory":"64gb", "walltime":"32:00:00", "account":"RNA_Unet", "cores":4} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 ../HotKnots/hotknots.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_ufold(files): 
    """
    Predict structure with Ufold
    """
    inputs = [files]
    outputs = [file.replace('data/test_files', 'steps/Ufold') for file in files]
    options = {"memory":"8gb", "walltime":"3:00:00", "account":"RNA_Unet"} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    echo "{files}" > input.txt

    CONDA_BASE=$(conda info --base)
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate UFold

    python3 ../UFOLD/ufold_predict.py
    mkdir steps/Ufold
    mv results_Ufold/* steps/Ufold/
    rm -r results_Ufold
    rm input.txt""".format(files = '\n'.join(files))
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_cnnfold(files): 
    """
    Predict structure with CNNfold
    """
    inputs = [file for file in files]
    outputs = [file.replace('data/test_files', 'steps/CNNfold') for file in files]
    options = {"memory":"16gb", "walltime":"18:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"

    python3 ../CNNfold/cnnfold_predict.py
    mv results_CNNfold/* steps/CNNfold/
    rm -r results_CNNfold"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_vienna(files): 
    """
    Predict structure with viennaRNA
    """
    inputs = [file for file in files]
    outputs = [file.replace('data/test_files', 'steps/vienna_mfold') for file in files]
    options = {"memory":"8gb", "walltime":"2:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    
    python3 other_methods/vienna_mfold.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_nussinov(files):
    """
    Predict structure with Nussinov algorithm
    """
    inputs = [file for file in files]
    outputs = [file.replace('data/test_files', 'steps/nussinov') for file in files] + ['results/times_nussinov.csv', 'figures/times_nussinov.png']
    options = {"memory":"8gb", "walltime":"96:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 other_methods/nussinov.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def evaluate_postprocessing(files): 
    """
    Evaluate all the implemented post-processing methods and compare them
    """
    inputs = [os.path.join('RNA_Unet.pth')] + files
    outputs = [os.path.join('results', 'evaluation_nn.csv'), #TODO - Fix paths to output
               os.path.join('figures', 'evaluation_nn.png')]
    options = {"memory":"16gb", "walltime":"24:00:00", "account":"RNA_Unet"} #NOTE - Think about memory and walltime
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/evaluate_postprocessing.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec) #TODO - Add some commands!

def compare_methods(methods_under600, files_under600, methods, files):
    """
    Compare the different previous methods with the RNAUnet
    """
    inputs = [file.replace('data/test_files', f'steps/{method}') for file in files_under600 for method in methods_under600] + [file.replace('data/test_files', f'steps/{method}') for file in files for method in methods]
    outputs = ['results/test_scores_under600.csv',
               'results/test_scores.csv',
               'results/f1_pseudoknots.csv',
               'results/average_scores_methods.csv',
               'results/RNAUnet_family_scores.csv'] #TODO - Add paths for plots
    options = {"memory":"16gb", "walltime":"24:00:00", "account":"RNA_Unet"} #NOTE - Think about memory and walltime
    spec = """echo "Job ID: $SLURM_JOB_ID\n
    python3 scripts/compare_methods.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec) 


### WORKFLOW ###
gwf = Workflow()

#Make data for experiments
for matrix_type in [8, 9, 17]:
    gwf.target_from_template(f'experiment_data_{matrix_type}', make_experiment_data(matrix_type))

#Make experiment of post processing time 
gwf.target_from_template('time_postprocess', postprocess_time())

#Make experiment of conversion time
gwf.target_from_template('time_convert', convert_time())


## FOR TRAINING THE ON THE ENTIRE DATA SET
gwf.target_from_template('convert_data', make_complete_set())

gwf.target_from_template('train_RNAUnet', train_model_small(files = pickle.load(open('data/train.pkl', 'rb')) + pickle.load(open('data/valid.pkl', 'rb'))))

gwf.target_from_template('evaluate_hotknots', evaluate_hotknots())



#Predicting with other methods for comparisons
under_600 = pickle.load(open('data/test_under_600.pkl', 'rb'))
excluded = [1282, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1297, 1298, 1299, 1300, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1914, 1915, 1908, 1909, 1910, 1911, 1912, 1901, 1902, 1903, 1904, 1905, 1906, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1868, 1865, 1866, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1718, 1719, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1311, 1312, 1313, 1314, 1315]
test_files = pickle.load(open('data/test.pkl', 'rb'))


files = [test_files[i] for i in under_600]
gwf.target_from_template('predict_hotknots', predict_hotknots(files))
gwf.target_from_template('predict_ufold', predict_ufold(files))

gwf.target_from_template('predict_cnnfold', predict_cnnfold(test_files))
gwf.target_from_template('predict_vienna', predict_vienna(test_files))
gwf.target_from_template('predict_nussinov', predict_nussinov(test_files))


methods_under600 = ['hotknots', 'Ufold']
methods = ['CNNfold', 'vienna_mfold', 'RNAUnet']

#gwf.target_from_template('compare_postprocessing', evaluate_postprocessing(test_files))
#gwf.target_from_template('evaluate_RNAUnet', test_model(test_files))
#gwf.target_from_template('compare_methods', compare_methods(methods_under600, files, methods, test_files))


