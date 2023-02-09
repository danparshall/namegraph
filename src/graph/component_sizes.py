import matplotlib.pyplot as plt
import networkit as nk
dir_name = '/home/juan.russy/shared/proof_run_FamNet/'
# base_filename = 'output/example.gml'
base_filename = 'output/Matched_Graph.txt'
output_filename = 'plots/component_size'

reader = nk.graphio.EdgeListReader(
    separator='\t', firstNode=0, continuous=False, directed=False)
G = reader.read(dir_name + base_filename)  # 'graph' object
# G = nk.readGraph(dir_name + base_filename, nk.graphio.Format.GML)
print('number of nodes', G.numberOfNodes(), 'number of edges', G.numberOfEdges())
print('is weighted', G.isWeighted())

cc = nk.components.ConnectedComponents(G)

# Run algorithm
cc.run()
# Extract the number of components (portions of the network that are disconnected from each other.)
print('number of components', cc.numberOfComponents())

# Get the component sizes in a dictionary
size = cc.getComponentSizes()
values = size.values()
values_list = sorted(list(values), reverse = True)
print('max', values_list[:10])  

# Number of components with sizes [2-5]
plt.figure(0)
plt.hist(values_list, edgecolor='black', bins=[
         1.5, 2.5, 3.5, 4.5, 5.5])
plt.title('Size of the components 2-5')
plt.savefig(dir_name + output_filename + '_2-5', dpi=200)

# Number of components with sizes [6-10]
plt.figure(1)
plt.hist(values_list, edgecolor='black', bins=[
         5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
plt.title('Size of the components 6-10')
plt.savefig(dir_name + output_filename + '_6-10', dpi=200)

# Number of components with sizes [11-100]
plt.figure(2)
plt.hist(values_list, edgecolor='black', bins=[
         10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5, 90.5, 100.5])
plt.title('Size of the components 11-100')
plt.savefig(dir_name + output_filename + '_11-100', dpi=200)

# Number of components with sizes [101-223]
plt.figure(3)
plt.hist(values_list, edgecolor='black', bins=[
         100.5, 120, 150, 224])
plt.title('Size of the components 101-223')
plt.savefig(dir_name + output_filename + '_101-223', dpi=200)
