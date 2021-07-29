import numpy as np
from matplotlib.cm import get_cmap
import cv2
import open3d as o3d

def get_color_palette(n):
    # For visualization
    # Note that COLOR_PALETTE is a [C, 3] np.uint8 array.
    cmap = get_cmap('rainbow', n)
    COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(n)])
    COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
    return COLOR_PALETTE

def get_seg_visualization(seg):
    # assume seg is (n, n)
    assert len(seg.shape) == 2
    COLOR_PALETTE = get_color_palette(np.max(seg)+1)
    img = COLOR_PALETTE[seg]
    return img

def show_image_by_opencv(rgb_img):
    cv2.imshow('image', rgb_img[:,:,::-1])
    cv2.waitKey(1)


def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def visualize_point_cloud(points, colors=None, normals=None,
                          show_frame=False, frame_size=1.0, frame_origin=(0, 0, 0)):
    """Visualize a point cloud."""
    pc = np2pcd(points, colors, normals)
    geometries = [pc]
    if show_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_origin)
        geometries.append(coord_frame)
    o3d.visualization.draw_geometries(geometries)