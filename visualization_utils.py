from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import numpy as np

def plot_mesh_3d(vertices, faces, elev=45, azim=30, filename=None):
    """
    Visualize mesh object. 
    """

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    pc = art3d.Poly3DCollection(vertices[faces],linewidths=0.1,edgecolor = 'k')
    ax.add_collection(pc)
    
    # ax.scatter(points[:,0], points[:,1], points[:,2], c='red', alpha = 0.2)
    
    # ax.set_xlim(-1.0, 1.0)
    # ax.set_ylim(-1.0, 1.0)
    # ax.set_zlim(-1.0, 1.0)
    
    # ax.set_title(, fontdict={'fontsize': 20, 'fontweight': 'medium'})
    ax.view_init(elev=elev, azim = azim)
    #print("storing ", filename)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    
def saveMeshPly(data, out_path):
    def toColor(c, vmin, vmax):
        inital = int(255 * (1 - (c - vmin) / (vmax - vmin)) )
        return max(min(255, inital), 0)
    
    # Generated point cloud, with colors:
    vertex = np.array([tuple(x) for x in data.x], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # Colors
    colors=np.copy(data.y[:,0]) # Pressure
    threshold = np.max(colors) #np.percentile(colors, 99.5)
    # vmin, vmax = np.min(colors), np.max(colors[colors < threshold])
    vmin, vmax = np.percentile(colors, 30), np.percentile(colors, 70)
    colors = [tuple( [toColor(c, vmin, vmax), 0, 0]) if c < threshold else tuple([0, 255, 0]) for c in colors]
    
    vertex_color = np.array(colors, 
                            dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

#     vertex_normals = np.array([tuple(x) for x in point_cloud_normals.tolist()],
#                               dtype=[('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

    # Merge the 3 tables together
    num_verts = len(vertex)
#     vertex_all = np.empty(num_verts, vertex.dtype.descr + vertex_color.dtype.descr + vertex_normals.dtype.descr)
    vertex_all = np.empty(num_verts, vertex.dtype.descr + vertex_color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in vertex_color.dtype.names:
        vertex_all[prop] = vertex_color[prop]

#     for prop in vertex_normals.dtype.names:
#         vertex_all[prop] = vertex_normals[prop]

    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(out_path)