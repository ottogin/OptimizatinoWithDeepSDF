import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

# from mesh_to_sdf import sample_sdf_near_surface

import trimesh
# import pyrender
import numpy as np

import json
import numpy as np
import time
import skimage.measure
import subprocess
import random
import pandas as pd
from tqdm import tqdm_notebook
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
from sklearn.neighbors import KDTree

#from mayavi import mlab
import plyfile
from pyntcloud import PyntCloud
from plyfile import PlyData

import deep_sdf
import deep_sdf.workspace as ws
from reconstruct import reconstruct


def load_model(experiment_directory, checkpoint):
    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    decoder.eval()
    
    return decoder

def create_mesh(
    decoder, latent_vec, filename='', N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    #print("sampling takes: %f" % (end - start))

    return convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def get_latent_from_mesh(decoder, latent_size, mesh_path='../Expirements/data/original_mesh.ply', 
                         num_iterations=500, num_samples=100):
    process_mesh(mesh_path, '../Expirements/data/original_SDF.npz', 
             'bin/PreprocessMesh', [])
    
    data_sdf = deep_sdf.data.read_sdf_samples_into_ram('../Expirements/data/original_SDF.npz')
#     data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
#     data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]
    
    err, latent = reconstruct(
                decoder,
                num_iterations,
                latent_size,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=num_samples,
                lr=5e-3,
                l2reg=True,
            )
    return latent
    
    
def get_latent_from_mesh_cpu(decoder, latent_size, mesh, 
                             num_iterations=500, num_samples=100):
    
    points, sdf = sample_sdf_near_surface(mesh)
    sdfs = np.hstack((points, sdf[:, None]))
    data_sdf = [torch.from_numpy(sdfs[sdfs[:, 3] > 0, :]), 
                torch.from_numpy(sdfs[sdfs[:, 3] < 0, :])]

    err, latent = reconstruct(
                decoder,
                num_iterations,
                latent_size,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=num_samples,
                lr=5e-3,
                l2reg=True,
            )
    return latent
    

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    norms_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])
        norms_tuple[i] = tuple(normals[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")
    el_norms = plyfile.PlyElement.describe(norms_tuple, "normals")

    ply_data = plyfile.PlyData([el_verts, el_faces, el_norms])
    return ply_data


def process_mesh(mesh_filepath, target_filepath, executable, additional_args):
    my_env = os.environ.copy()
    my_env["PANGOLIN_WINDOW_URI"] = "headless://"
    
    command = [executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args

    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=my_env)
    subproc.wait()
    
    out, err = subproc.communicate()
    if out is not None:
        print(out.decode())
    if err is not None:
        print(err.decode())
        

def plot_mesh_from_vector(decoder, initial_latent, N=256):
    ply_mesh = None

    with torch.no_grad():
        ply_mesh = create_mesh( decoder,
                                initial_latent,
                                N=N,
                                max_batch=int(2 ** 18),
                                offset=None,
                                scale=None)
    ply_mesh.write('../Expirements/data/original_mesh.ply')
    
    cloud = PyntCloud.from_file('../Expirements/data/original_mesh.ply')
    cloud.plot(background='white', initial_point_size=0.003)
    
    

def get_trimesh_from_torch_geo_with_colors(mesh, preds, vmin=-8, vmax=8):
    norm = mpl.colors.Normalize(vmin= vmin, vmax=vmax)
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    verticies = mesh.x.cpu().detach()
    faces = mesh.face.t().cpu().detach()
    return trimesh.Trimesh(vertices=verticies, faces=faces, 
                           vertex_colors=list(map(lambda c: m.to_rgba(c),  preds[:, 0].cpu().detach())))
    
    
def get_trimesh_from_ply_with_colors(ply_mesh, preds):
    norm = mpl.colors.Normalize(vmin= -8, vmax=8)
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    verticies = np.array([ply_mesh['vertex']['x'], ply_mesh['vertex']['y'], ply_mesh['vertex']['z']]).transpose()
    faces = ply_mesh['face']['vertex_indices']
    return trimesh.Trimesh(vertices=verticies, faces=faces, 
                           vertex_colors=list(map(lambda c: m.to_rgba(c),  preds[:, 0].cpu().detach())))
    
    
def get_trimesh_from_latent(decoder, latent, N=256):
    with torch.no_grad():
        ply_mesh = create_mesh( decoder,
                            latent,
                            N=N,
                            max_batch=int(2 ** 18),
                            offset=None,
                            scale=None)
    verticies = np.array([ply_mesh['vertex']['x'], ply_mesh['vertex']['y'], ply_mesh['vertex']['z']]).transpose()
    faces = ply_mesh['face']['vertex_indices']
    return trimesh.Trimesh(vertices=verticies, faces=faces)

def get_cloud_from_latent(decoder, initial_latent, N=256, save_path=None):
    ply_mesh = None

    with torch.no_grad():
        ply_mesh = create_mesh( decoder,
                                initial_latent,
                                N=N,
                                max_batch=int(2 ** 18),
                                offset=None,
                                scale=None)
        
    if save_path is None:
        save_path = '../Expirements/data/original_mesh.ply'

    ply_mesh.write(save_path)
    
    cloud = PyntCloud.from_file(save_path)
    return cloud

def subsample_from_torch_tensor(tensor, num_points):
    if num_points < tensor.size(0):
        return tensor[torch.randperm(tensor.size(0))[:num_points]]
    else:
        return tensor

def get_points_from_latent(decoder, initial_latent, N=256, point_num=None, save_path=None):
    cloud = get_cloud_from_latent(decoder, initial_latent, N, save_path=save_path)
    nparr = cloud.points.to_numpy()
    if point_num is not None:
        nparr = random.sample(list(nparr), point_num)
    return torch.cuda.FloatTensor(nparr)


def plot_points_from_torch(tensor):
    cloud = PyntCloud(pd.DataFrame(tensor, columns=['x', 'y', 'z']))
    cloud.plot(background='white', initial_point_size=0.003)
    
def chamfer_distance_without_batch(p1, p2, debug=False):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    if debug:
        print(p1[0][0])

    p1 = p1.repeat(p2.size(1), 1, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(0, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print(p1[0])

    p2 = p2.repeat(p1.size(0), 1, 1)
    if debug:
        print('p2 size is {}'.format(p2.size()))
        print(p2[0])

    dist = torch.add(p1, torch.neg(p2))
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=2)
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.min(dist, dim=1)[0]
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.sum(dist)
    if debug:
        print('-------')
        print(dist)

    return dist


def chamfer_distance_with_batch(p1, p2, debug):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    if debug:
        print(p1[0])

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print('p2 size is {}'.format(p2.size()))
        print(p1[0][0])

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(1, 2)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print(p1[0][0])

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    if debug:
        print('p2 size is {}'.format(p2.size()))
        print(p2[0][0])

    dist = torch.add(p1, torch.neg(p2))
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=3)
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.min(dist, dim=2)[0]
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.sum(dist)
    if debug:
        print('-------')
        print(dist)

    return dist