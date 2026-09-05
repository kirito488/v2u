import os
import open3d as o3d
import numpy as np 

from mvp.data.util import rotation_matrix, bbox_shift, bbox_rotate
from mvp.config import model_3d_path, model_3d_examples


def get_model_mesh(model_3d_name, bbox):
    mesh = o3d.io.read_triangle_mesh(os.path.join(model_3d_path, "{}.ply".format(model_3d_name)))
    model_bbox = np.array(model_3d_examples[model_3d_name])
    translate = bbox[:3]
    rotate = bbox[6]
    scale = np.min(bbox[3:6] / model_bbox[3:6]) 
    if scale is not None:
        mesh.scale(scale, np.array([0, 0, 0]).T)
    if rotate is not None:
        mesh.rotate(rotation_matrix(0, rotate, 0), np.zeros(3).T)
    if translate is not None:
        mesh.translate(np.array([translate[0], translate[1], translate[2]]).T)
    return mesh


def get_wall_mesh(bbox):
    wall = o3d.geometry.TriangleMesh.create_box(width=bbox[3], height=bbox[4], depth=bbox[5])
    wall.translate(np.array([-bbox[3]/2, -bbox[4]/2, 0]).T)
    wall.rotate(rotation_matrix(0, bbox[6], 0), np.zeros(3).T)
    wall.translate(np.array([bbox[0], bbox[1], bbox[2]]).T)
    return wall


def get_box_mesh(bbox):
    """用长方体代替缺失的车辆 ply，尺寸/朝向与 [x,y,z,l,w,h,yaw] 一致。"""
    return get_wall_mesh(bbox)


def get_model_bbox(model_3d_name, bbox):
    model_bbox = np.array(model_3d_examples[model_3d_name])
    translate = bbox[:3]
    rotate = bbox[6]
    scale = np.min(bbox[3:6] / model_bbox[3:6])
    bbox = model_bbox
    if scale is not None:
        bbox[3:6] *= scale
    if rotate is not None:
        bbox = bbox_rotate(bbox, np.array([0, float(rotate), 0]))
    if translate is not None:
        bbox = bbox_shift(bbox, np.array([translate[0], translate[1], translate[2]]))
    return bbox


def _ray_intersection_o3d(meshes, rays):
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id_map = {}
    for mesh in meshes:
        mesh_cuda = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_id = scene.add_triangles(mesh_cuda)
        mesh_id_map[mesh_id] = mesh

    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ray_size = rays_t.shape[0]
    ans_raw = scene.cast_rays(rays_t)
    ans = {key: ans_raw[key].numpy() for key in ans_raw}

    intersection = np.zeros((ray_size, 3))
    for i in range(ray_size):
        data = {key: value[i] for key, value in ans.items()}
        if data["t_hit"] > 10000:
            intersection[i] = np.array([np.inf, np.inf, np.inf])
        else:
            mesh = mesh_id_map[data["geometry_ids"]]
            triangle_vertices = mesh.triangles[data["primitive_ids"]]
            intersection[i] = (
                (1 - np.sum(data["primitive_uvs"])) * mesh.vertices[triangle_vertices[0]]
                + data["primitive_uvs"][0] * mesh.vertices[triangle_vertices[1]]
                + data["primitive_uvs"][1] * mesh.vertices[triangle_vertices[2]]
            )
    return intersection


def _mesh_triangles(meshes):
    verts_list, tris_list = [], []
    offset = 0
    for mesh in meshes:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        t = np.asarray(mesh.triangles, dtype=np.int64)
        if v.size == 0 or t.size == 0:
            continue
        verts_list.append(v)
        tris_list.append(t + offset)
        offset += v.shape[0]
    if not verts_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(verts_list), np.vstack(tris_list)


def _ray_intersection_numpy(meshes, rays):
    """Möller–Trumbore，不依赖 Open3D RaycastingScene。"""
    rays = np.asarray(rays, dtype=np.float64)
    n = rays.shape[0]
    origins = rays[:, :3]
    dirs = rays[:, 3:6]
    dn = np.linalg.norm(dirs, axis=1, keepdims=True)
    dn = np.maximum(dn, 1e-12)
    dirs = dirs / dn

    verts, tris = _mesh_triangles(meshes)
    out = np.full((n, 3), np.inf, dtype=np.float64)
    if tris.shape[0] == 0:
        return out

    best_t = np.full(n, np.inf, dtype=np.float64)
    eps = 1e-8
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    for ti in range(tris.shape[0]):
        e1 = v1[ti] - v0[ti]
        e2 = v2[ti] - v0[ti]
        h = np.cross(dirs, e2)
        a = np.einsum("ij,j->i", h, e1)
        hit = np.abs(a) > eps
        if not np.any(hit):
            continue
        f = np.zeros(n, dtype=np.float64)
        f[hit] = 1.0 / a[hit]
        s = origins - v0[ti]
        u = f * np.einsum("ij,ij->i", s, h)
        hit &= (u >= 0.0) & (u <= 1.0)
        if not np.any(hit):
            continue
        q = np.cross(s, e1)
        vv = f * np.einsum("ij,ij->i", q, dirs)
        hit &= (vv >= 0.0) & (u + vv <= 1.0)
        if not np.any(hit):
            continue
        t = f * np.einsum("ij,j->i", q, e2)
        hit &= t > eps
        closer = hit & (t < best_t)
        if not np.any(closer):
            continue
        best_t[closer] = t[closer]
        out[closer] = origins[closer] + dirs[closer] * t[closer, None]
    return out


def ray_intersection(meshes, rays):
    if hasattr(o3d.t.geometry, "RaycastingScene"):
        try:
            return _ray_intersection_o3d(meshes, rays)
        except Exception:
            pass
    return _ray_intersection_numpy(meshes, rays)
