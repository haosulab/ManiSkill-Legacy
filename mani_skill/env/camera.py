import numpy as np, os

def get_texture_by_dltensor(cam, tag, dtype='float'):
    if os.environ.get('NO_CUPY', 0) == 1:
        from cupy import fromDlpack, asnumpy
        dlpack = cam.get_dl_tensor(tag)
        return asnumpy(fromDlpack(dlpack))
    else:
        return getattr(cam, f'get_{dtype}_texture')(tag)

def read_images_from_camera(cam, depth=False, seg_indices=None):
    img = {}
    img['rgb'] = get_texture_by_dltensor(cam, "Color")[:, :, :3]
    if depth:
        img['depth'] = get_texture_by_dltensor(cam, "Position")[:, :, [-1]]
    if seg_indices is not None:
        seg = get_texture_by_dltensor(cam, "Segmentation", 'uint32')[..., seg_indices] # 0 is visual id, 1 is actor id
        if len(seg.shape) < 3:
            seg = np.expand_dims(seg, axis=2)
        img['seg'] = seg
    return img

def read_pointclouds_from_camera_with_depth_filtering(cam, depth=True, seg_indices=None):
    # the depth parameter will be ignored
    pcd = {}
    pos_depth = get_texture_by_dltensor(cam, "Position")
    mask = pos_depth[..., -1] < 1
    pcd['rgb'] = get_texture_by_dltensor(cam, "Color")[:, :, :3][mask]
    pcd['xyz'] = pos_depth[:, :, :3][mask]
    if seg_indices is not None:
        seg = get_texture_by_dltensor(cam, "Segmentation", 'uint32')[..., seg_indices]
        if len(seg.shape) < 3:
            seg = np.expand_dims(seg, axis=2)
        pcd['seg'] = seg[mask]
    return pcd

def read_pointclouds_from_camera(cam, depth=True, seg_indices=None):
    # the depth parameter will be ignored
    pcd = {}
    pos_depth = get_texture_by_dltensor(cam, "Position")
    # here we assume all pos_depth[..., -1] < 1
    pcd['rgb'] = get_texture_by_dltensor(cam, "Color")[:, :, :3].reshape(-1, 3)
    pcd['xyz'] = pos_depth[:, :, :3].reshape(-1, 3)
    if seg_indices is not None:
        seg = get_texture_by_dltensor(cam, "Segmentation", 'uint32')[..., seg_indices]
        if len(seg.shape) < 3:
            seg = np.expand_dims(seg, axis=2)
        pcd['seg'] = seg.reshape(pcd['xyz'].shape[0], -1)
    return pcd

class CombinedCamera(object):
    def __init__(self, name, sub_cameras):
        self.name = name
        self.sub_cameras = sub_cameras
    def get_name(self):
        return self.name

    def take_picture(self):
        for cam in self.sub_cameras:
            cam.take_picture()

    def get_model_matrix(self):
        return np.eye(4)

    def get_images_list(self, depth=False, seg_indices=None):
        view_list = []
        for cam in self.sub_cameras:
            view_dict = read_images_from_camera(cam, depth, seg_indices)
            view_list.append(view_dict)
        return view_list

    def get_fused_pointcloud(self, seg_indices=None):
        pcds = []
        for cam in self.sub_cameras:
            pcd = read_pointclouds_from_camera(cam, seg_indices=seg_indices) # dict
            pcds.append(pcd)
        for i, pcd in enumerate(pcds):
            T = self.sub_cameras[i].get_model_matrix()
            R = T[:3, :3]
            t = T[:3, 3]
            pcd['xyz'] = pcd['xyz'] @ R.transpose() + t
        fused_pcd = {}
        for key in pcds[0].keys():
            fused_pcd[key] = np.concatenate([pcd[key] for pcd in pcds], axis=0)
        return fused_pcd

    def get_combined_view(self, mode, depth=False, seg_indices=None):
        if mode == 'color_image':
            return self.get_images_list(depth, seg_indices)
        else: # pointcloud
            return self.get_fused_pointcloud(seg_indices)
