import pathlib
import networkit as nk
dir_name = '/home/juan.russy/shared/proof_run_FamNet/interim/'
base_filename = 'PROOF_GRAPH_matched_exact_name.txt'
filename = dir_name + base_filename
print(filename)

nk.engineering.setNumberOfThreads(2)
g = nk.graphio.EdgeListReader(separator = '\t', firstNode = 0, continuous = True, directed = True).read(filename)
nk.graphio.writeGraph(g, dir_name + "example.gml", nk.Format.GML)
