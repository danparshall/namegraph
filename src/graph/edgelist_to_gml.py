import time
import networkit as nk
dir_name = ''
base_filename = 'PROOF_GRAPH_matched_exact_name.txt'
filename = dir_name + base_filename
print(filename)

startTime = time.time()

nk.engineering.setNumberOfThreads(2)
g = nk.graphio.EdgeListReader(separator = '\t', firstNode = 0, continuous = False, directed = False).read(filename)
nk.graphio.writeGraph(g, dir_name + "example.gml", nk.Format.GML)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
