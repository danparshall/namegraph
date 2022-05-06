import networkit as nk
dir_name = ''
base_filename = 'example.gml'

example = nk.readGraph(dir_name + base_filename, nk.graphio.Format.GML)

nk.viztasks.drawGraph(example)
