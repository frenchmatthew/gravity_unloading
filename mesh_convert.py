import meshio 
import pyvista as pv  
import numpy as np 

def mesh_convert(path, output="linearise.vtu", fixed_nodes=None):  
    mesh = pv.read(path)  
    mesh["fixed_nodes"] = np.zeros(mesh.n_points) 
    coordinates = mesh.points 
    for i, node in enumerate(fixed_nodes): 
        node_idx = np.argmin(np.linalg.norm(coordinates - node, axis=1)) 
        mesh["fixed_nodes"][node_idx] = 1.0
    linear_mesh = mesh.tessellate(max_n_subdivide=3)
    linear_mesh.save(output) 
    mesh = meshio.read(output)   
    return mesh 

if __name__ == '__main__':  

    fixed_nodes = np.load("example.npy")
    mesh = mesh_convert( 
        path="example.vtu", # cubic Lagrange or abitrary order mesh
        output="linearise.vtu", # linear mesh
        fixed_nodes=fixed_nodes
    ) 
    mesh.save("example.xdmf")

