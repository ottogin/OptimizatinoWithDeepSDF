
import pdb
import numpy as np
import torch
import neural_renderer as nr
import os
import imageio
import trimesh
import pymesh
import argparse
from scipy.spatial import cKDTree as KDTree
import pdb

def get_rotate_matrix(rotation_angle1):
    cosval = np.cos(rotation_angle1)
    sinval = np.sin(rotation_angle1)

    rotation_matrix_x = np.array([[1, 0, 0, 0],
                                  [0, cosval, -sinval, 0],
                                  [0, sinval, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_y = np.array([[cosval, 0, sinval, 0],
                                  [0, 1, 0, 0],
                                  [-sinval, 0, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                  [sinval, cosval, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    scale_y_neg = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    neg = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    # y,z swap = x rotate -90, scale y -1
    # new_pts0[:, 1] = new_pts[:, 2]
    # new_pts0[:, 2] = new_pts[:, 1]
    #
    # x y swap + negative = z rotate -90, scale y -1
    # new_pts0[:, 0] = - new_pts0[:, 1] = - new_pts[:, 2]
    # new_pts0[:, 1] = - new_pts[:, 0]

    # return np.linalg.multi_dot([rotation_matrix_z, rotation_matrix_y, rotation_matrix_y, scale_y_neg, rotation_matrix_z, scale_y_neg, rotation_matrix_x])
    return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])

def get_projection_matricies(az, el, distance_ratio, roll = 0, focal_length=35, img_w=137, img_h=137):
    """
    Calculate 4x3 3D to 2D projection matrix given viewpoint parameters.
    Code from "https://github.com/Xharlie/DISN"
    """

    F_MM = focal_length  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    CAM_MAX_DIST = 1.75
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                      [1.0, -4.371138828673793e-08, -0.0],
                      [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    # Calculate intrinsic matrix.
    scale = RESOLUTION_PCT / 100
    # print('scale', scale)
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    # print('f_u', f_u, 'f_v', f_v)
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))
    # finally, consider roll
    cr = np.cos(np.radians(roll))
    sr = np.sin(np.radians(roll))
    R_z = np.matrix(((cr, -sr, 0),
                  (sr, cr, 0),
                  (0, 0, 1)))

    rot_mat = get_rotate_matrix(-np.pi / 2)

    return K, R_z@RT@rot_mat


def visualize_constraint(vertices, faces, img_resolution = 1200, azimuth = -90, elevation = 0, distance_ratio = 1.0):
    """
    Interface to neural_render to produce nice visualizations. It requires GPU.
    Inputs:
    vertices in [V,3]
    faces in [F,3]
    Output:
    Image in [img_resolution, img_resolution, 3]
    """

    # first set up camera
    intrinsic, extrinsic =  get_projection_matricies(azimuth, elevation, distance_ratio, img_w=img_resolution, img_h=img_resolution)

    K_cuda = torch.tensor(intrinsic[np.newaxis, :, :].copy()).float().cuda().unsqueeze(0)
    R_cuda = torch.tensor(extrinsic[np.newaxis, 0:3, 0:3].copy()).float().cuda().unsqueeze(0)
    t_cuda = torch.tensor(extrinsic[np.newaxis, np.newaxis, 0:3, 3].copy()).float().cuda().unsqueeze(0)

    # initialize renderer
    renderer = nr.Renderer(image_size = img_resolution, orig_size = img_resolution, K=K_cuda, R=R_cuda, t=t_cuda, anti_aliasing=True)

    # now move vertices, faces to GPU
    verts_dr = torch.tensor(vertices.copy(), dtype=torch.float32, requires_grad = False).cuda()
    faces_dr = torch.tensor(faces.copy()).cuda()
    textures_dr = torch.ones(faces_dr.shape[0], 1, 1, 1, 3, dtype=torch.float32).cuda()
    textures_dr[:,:,:,:,1:3]=0.1


    images_out, _, alpha_export = renderer(verts_dr.unsqueeze(0), faces_dr.unsqueeze(0), textures_dr.unsqueeze(0))
    image_out_export = 255*images_out.detach().cpu().numpy()[0].transpose((1, 2, 0))
    alpha_ou_export = 255*alpha_export.detach().cpu().numpy()[0][:,:,np.newaxis]

    return np.concatenate((image_out_export, alpha_ou_export), -1)

def visualize_car(vertices, faces, img_resolution = 1200, azimuth = -90, elevation = 0, distance_ratio = 1.0):
    """
    Interface to neural_render to produce nice visualizations. It requires GPU.
    Inputs:
    vertices in [V,3]
    faces in [F,3]
    Output:
    Image in [img_resolution, img_resolution, 3]
    """

    # first set up camera
    intrinsic, extrinsic =  get_projection_matricies(azimuth, elevation, distance_ratio, img_w=img_resolution, img_h=img_resolution)

    K_cuda = torch.tensor(intrinsic[np.newaxis, :, :].copy()).float().cuda().unsqueeze(0)
    R_cuda = torch.tensor(extrinsic[np.newaxis, 0:3, 0:3].copy()).float().cuda().unsqueeze(0)
    t_cuda = torch.tensor(extrinsic[np.newaxis, np.newaxis, 0:3, 3].copy()).float().cuda().unsqueeze(0)

    # initialize renderer
    renderer = nr.Renderer(image_size = img_resolution, orig_size = img_resolution, K=K_cuda, R=R_cuda, t=t_cuda, anti_aliasing=True)

    # now move vertices, faces to GPU
    verts_dr = torch.tensor(vertices.copy(), dtype=torch.float32, requires_grad = False).cuda()
    faces_dr = torch.tensor(faces.copy()).cuda()
    textures_dr = torch.ones(faces_dr.shape[0], 1, 1, 1, 3, dtype=torch.float32).cuda()

    images_out, _, alpha_export = renderer(verts_dr.unsqueeze(0), faces_dr.unsqueeze(0), textures_dr.unsqueeze(0))
    image_out_export = 255*images_out.detach().cpu().numpy()[0].transpose((1, 2, 0))
    alpha_ou_export = 255*alpha_export.detach().cpu().numpy()[0][:,:,np.newaxis]

    return np.concatenate((image_out_export, alpha_ou_export), -1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate an image of constraints')
    
    parser.add_argument('inp', metavar='inp', type=str,
                        help='Path to input shape')
    parser.add_argument('out', metavar='out', type=str,
                        help='Name of the output')
    
    args = parser.parse_args()
    
    # by default in [0,0,0] with radius 1
    constraint = pymesh.generate_icosphere(1, [0, 0, 0], 4)  # 2562 vertices
    # Sphere with R=0.2 in (-0.05, 0.05, 0)
    vertices_1 = constraint.vertices*0.11 + np.array([0, 0.05, 0])
    # Sphere with R=0.1 in (0.3, 0, 0)
    vertices_2 = constraint.vertices*0.088 + np.array([-0.4, 0, 0])

    all_verts = np.concatenate((vertices_1, vertices_2),0)
    all_faces = np.concatenate((constraint.faces, constraint.faces+constraint.vertices.shape[0]),0)

    image_constraints = visualize_constraint(all_verts, all_faces)

    image_filename = os.path.join("./constraint.png")
    imageio.imwrite(image_filename, image_constraints.astype(np.uint8))

    car = trimesh.load(args.inp)
    image_car = visualize_car(car.vertices , car.faces) #+ [-0.07, 0, 0]

    image_filename = os.path.join("./car.png")
#     imageio.imwrite(image_filename, image_car.astype(np.uint8))

    mask_constraint = image_constraints[:,:,3]>0

    alpha = 0.4
    image_car[mask_constraint, :]= alpha * image_constraints[mask_constraint, :] +\
                                   (1 - alpha) * image_car[mask_constraint, :]
    image_filename = os.path.join(args.out + ".png")
    imageio.imwrite(image_filename, image_car.astype(np.uint8))

    print("Done.")
