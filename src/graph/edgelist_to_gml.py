import time
import networkit as nk
dir_name = '/home/juan.russy/shared/proof_run_FamNet/output/'
base_filename = 'Matched_Graph.txt'
output_filename = 'example.gml'
filename = dir_name + base_filename
print(filename)

#%% EdgeListReader method
# EdgeListReader -> Execution time in seconds: 154.35551619529724
# writeGraph -> Execution time in seconds: 20.2667133808136
# nk.engineering.setNumberOfThreads(2)
# startTime = time.time()
# g = nk.graphio.EdgeListReader(separator = '\t', firstNode = 0, continuous = False, directed = False).read(filename)
# executionTime = (time.time() - startTime)
# print('EdgeListReader -> Execution time in seconds: ' + str(executionTime))

# startTime = time.time()
# nk.graphio.writeGraph(g, dir_name + output_filename, nk.Format.GML)
# executionTime = (time.time() - startTime)
# print('writeGraph -> Execution time in seconds: ' + str(executionTime))

#%% GMLGraphReader
startTime = time.time()
g = nk.graphio.GMLGraphReader().read(dir_name + output_filename)
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
