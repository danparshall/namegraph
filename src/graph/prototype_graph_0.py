import pathlib
import networkit as nk
dir_name = ''
base_filename = ""
filename = dir_name + base_filename
print(filename)
# with open(dir_name + base_filename) as f:
#     lines = f.readlines()
# print(lines[:10])

nk.engineering.setNumberOfThreads(2)
g = nk.graphio.EdgeListReader(separator = '\t', firstNode = 0, continuous = True, directed = True).read(filename)
nk.graphio.writeGraph(g, dir_name + "example.gml", nk.Format.GML)
