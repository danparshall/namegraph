import networkit as nk
import pandas as pd

# Paths
dir_name = ''
base_filename = ''  # EdgeList File
filename = dir_name + base_filename
print(filename)

print('Read EdgeList')
reader = nk.graphio.EdgeListReader(
    separator='\t', firstNode=0, continuous=False, directed=False)
G = reader.read(filename)  # 'graph' object

def Map_IDnk_cedula(reader):
    """ Obtain the mapping between networkit ID and Cedula

    Networkit automatically changes node ID's for integers between
    1 and N-1. This function gives you a dataframe with this
    information.

    Args:
        reader: 'graph' object from networkit
    Returns:
        map_df: Mapping dataframe  
    """
    map_nkID_ced = reader.getNodeMap()
    # print('len map', len(map_nkID_ced.keys()))
    map_df = pd.DataFrame.from_dict(map_nkID_ced, orient='index')
    map_df = map_df.reset_index(level=0)
    map_df.columns = ['id', 'nk_id']
    return map_df

print(Map_IDnk_cedula(reader).head())