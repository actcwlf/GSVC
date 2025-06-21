import math
from dataclasses import dataclass
from typing import NamedTuple
import torch.nn as nn
import torch


BLOCK_X = 16
BLOCK_Y = 16


def mark_visible(positions, view_matrix, proj_matrix):
    pass


@dataclass
class float4:
    x: float
    y: float
    z: float
    w: float


@dataclass
class float3:
    x: float
    y: float
    z: float


@dataclass
class float2:
    x: float
    y: float


@dataclass
class int2:
    x: int
    y: int


@dataclass
class dim3:
    x: int
    y: int
    z: int


def transformPoint4x4(p, matrix):
    transformed = float4(
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    )
    return transformed


def transformPoint4x3(p, matrix):
    transformed = float3(
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    )
    return transformed


def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5



def computeCov3D(scale, mod, rot):
    # // Create scaling matrix
    # glm::mat3 似乎应等价为 torch.eye
    S = torch.eye(3)

    S[0, 0] = mod * scale.x
    S[1, 1] = mod * scale.y
    S[2, 2] = mod * scale.z

    # // Normalize quaternion to get valid rotation
    q = rot   #;// / glm::length(rot);
    # rotation 矩阵中第一列是齐次项，因此要进行一下转换
    r = q.x
    x = q.y
    y = q.z
    z = q.w

    # // Compute rotation matrix from quaternion
    # 原始情况似乎是一个左手系的转换，但论文里又似乎使用的是右手系的推导
    # glm::mat3 R = glm::mat3(
    # 	1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
    # 	2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
    # 	2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    # );

    R = torch.tensor([
        [1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y)],
        [2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x)],
        [2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y)]
    ])

    M: torch.Tensor = S @ R

    # // Compute 3D world covariance matrix Sigma
    Sigma = M.permute(1, 0) @ M

    # // Covariance is symmetric, only store upper right
    cov3D = torch.Tensor([0, 0, 0, 0, 0, 0])
    cov3D[0] = Sigma[0, 0]
    cov3D[1] = Sigma[0, 1]
    cov3D[2] = Sigma[0, 2]
    cov3D[3] = Sigma[1, 1]
    cov3D[4] = Sigma[1, 2]
    cov3D[5] = Sigma[2, 2]
    return cov3D


def in_frustum(orig_points, view_matrix, proj_matrix, prefiltered):
    p_orig = float3(*orig_points)

    #  Bring points to screen space
    p_hom = transformPoint4x4(p_orig, proj_matrix)
    p_w = 1.0 / (p_hom.w + 0.0000001)
    p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w)
    p_view = transformPoint4x3(p_orig, view_matrix)

    if p_view.z <= 0.2:  # // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
        if prefiltered:
            raise ValueError('Trap')

        return False

    return True


def in_frustum_with_depth(orig_points, view_matrix, proj_matrix, prefiltered):
    p_orig = float3(*orig_points)

    #  Bring points to screen space
    # p_hom = transformPoint4x4(p_orig, proj_matrix)
    # p_w = 1.0 / (p_hom.w + 0.0000001)
    # p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w)
    p_view = transformPoint4x3(p_orig, view_matrix)

    if p_view.z <= 0.2:  # // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
        if prefiltered:
            raise ValueError('Trap')

        return False, None

    return True, p_view.z


def in_ortho_with_depth(orig_points, view_matrix, prefiltered):
    p_orig = float3(*orig_points)

    #  Bring points to screen space
    # p_hom = transformPoint4x4(p_orig, proj_matrix)
    # p_w = 1.0 / (p_hom.w + 0.0000001)
    # p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w)
    p_view = transformPoint4x3(p_orig, view_matrix)

    if p_view.z < -300:  # // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
        if prefiltered:
            raise ValueError('Trap')

        return False, None

    return True, p_view.z


def computeCov2D(
        mean,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
        cov3D,
        viewmatrix):

    # // The following models the steps outlined by equations 29
	# // and 31 in "EWA Splatting" (Zwicker et al., 2002).
	# // Additionally considers aspect / scaling of viewport.
	# // Transposes used to account for row-/column-major conventions.
    t = transformPoint4x3(mean, viewmatrix)

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t.x / t.z
    tytz = t.y / t.z
    t.x = min(limx, max(-limx, txtz)) * t.z
    t.y = min(limy, max(-limy, tytz)) * t.z

    # // 这里只计算投影变换的jaccobian mat，不需要投影矩阵本身
    # // 这里的投影变换似乎包含了从屏幕空间到像素空间的缩放
    # glm::mat3 J = glm::mat3(
    # 	focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
    # 	0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
    # 	0, 0, 0);

    J = torch.Tensor([
    	[focal_x / t.z, 0.0,           -(focal_x * t.x) / (t.z * t.z)],
    	[0.0,           focal_y / t.z, -(focal_y * t.y) / (t.z * t.z)],
    	[0,             0,             0]
    ]).permute(1, 0)

    # glm::mat3 W = glm::mat3(
    # 	viewmatrix[0], viewmatrix[4], viewmatrix[8],
    # 	viewmatrix[1], viewmatrix[5], viewmatrix[9],
    # 	viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    W = torch.Tensor([
    	[viewmatrix[0], viewmatrix[4], viewmatrix[8]],
    	[viewmatrix[1], viewmatrix[5], viewmatrix[9]],
    	[viewmatrix[2], viewmatrix[6], viewmatrix[10]]

    ]).permute(1, 0)

    T = W @ J

    Vrk = torch.tensor([
        [cov3D[0], cov3D[1], cov3D[2]],
        [cov3D[1], cov3D[3], cov3D[4]],
        [cov3D[2], cov3D[4], cov3D[5]]
    ]).permute(1, 0)

    cov = T.permute(1, 0) @ Vrk.permute(1, 0) @ T

    # // Apply low-pass filter: every Gaussian should be at least
    # // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3
    cov[1][1] += 0.3
    return float3( float(cov[0][0]), float(cov[0][1]), float(cov[1][1]))


def getRect(p, max_radius, grid):
    # BLOCK代表的时tile数量
    rect_min = int2(
        min(grid.x, max(0, int((p.x - max_radius) / BLOCK_X))),
        min(grid.y, max(0, int((p.y - max_radius) / BLOCK_Y)))
    )  # 扣除gaussian 投影后构成矩形区域，对此区域划分tile后，tile的大小
    rect_max = int2(
        min(grid.x, max(0, int((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max(0, int((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
    ) # 包含gaussian 投影后的矩形区域，划分tile后，tile的大小
    return rect_min, rect_max


def ndc_culling(p_orig: float3, proj_matrix, image_width, image_height, radius=1):
    grid = dim3((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1)

    # p_orig = float3(*orig_points)
    p_hom = transformPoint4x4(p_orig, proj_matrix)
    p_w = 1.0 / (p_hom.w + 0.0000001)
    p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w) # 在投影空间中的坐标

    # 正常情况下，投影变换后得到的剪切空间里，x, y, z值 (-1, -1, 0) (1, 1, 1)之间，可能因使用的图形库存在差异
    # 转换到ndc坐标时，可能会出现右手系到左手系的转换
    point_image = float2(ndc2Pix(p_proj.x, image_width), ndc2Pix(p_proj.y, image_height))
    # uint2 rect_min, rect_max;

    rect_min, rect_max = getRect(point_image, radius, grid)
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0):
        return False
    return True



def rasterize_gaussians_filter(
        means3D,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        view_matrix,
        proj_matrix,
        tan_fovx,
        tan_fovy,
        image_height,
        image_width,
        prefiltered,
        debug):

    grid = dim3((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1)

    focal_y = image_height / (2.0 * tan_fovy)

    focal_x = image_width / (2.0 * tan_fovx)
    radii = []
    for idx in range(1000): # means3D.shape[0]):
        orig_points = means3D[idx, :]

        radius = 0
        if not in_frustum(orig_points, view_matrix, proj_matrix, prefiltered):
            radii.append(radius) # radii[idx] == 0
            continue

        p_orig = float3(*orig_points)
        p_hom = transformPoint4x4(p_orig, proj_matrix)
        p_w = 1.0 / (p_hom.w + 0.0000001)
        p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w)

        # assert cov3D_precomp is None  # 暂时不允许预计算 / 直接丢弃预计算的covD

        scale = float3(*scales[idx, :])
        rot = float4(*rotations[idx, :])

        cov3D = computeCov3D(scale, scale_modifier, rot)

        cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, view_matrix)

        # // Invert covariance(EWA algorithm)
        det = (cov.x * cov.z - cov.y * cov.y)
        if det == 0.0:
            radii.append(radius)   # radii[idx] == 0
            continue

        # // Compute extent in screen space(by finding eigenvalues of
        # // 2D covariance matrix).Use extent to compute a bounding rectangle
        # // of screen - space tiles that this Gaussian overlaps with.Quit if
        # // rectangle covers 0 tiles.

        mid = 0.5 * (cov.x + cov.z)
        lambda1 = mid + math.sqrt(max(0.1, mid * mid - det))
        lambda2 = mid - math.sqrt(max(0.1, mid * mid - det))
        my_radius = math.ceil(3. * math.sqrt(max(lambda1, lambda2))) # 已经是在像素空间的长度
        point_image = float2(ndc2Pix(p_proj.x, image_width), ndc2Pix(p_proj.y, image_height))
        # uint2 rect_min, rect_max;

        rect_min, rect_max = getRect(point_image, my_radius, grid)
        if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0):
            radii.append(radius)   # radii[idx] == 0
            continue

        radii.append(my_radius)

    return torch.Tensor(radii)









class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None,
                cov3D_precomp=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
                (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

    def visible_filter(self, means3D, scales=None, rotations=None, cov3D_precomp=None):

        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # print('means3D\n', means3D[64:65, ...])
        # print('scales\n', scales[64:65, ...])
        # print('rotations\n', rotations[64:65, ...])
        # print('scale_modifier', raster_settings.scale_modifier)
        # print(cov3D_precomp)
        # print('vm\n', raster_settings.viewmatrix)
        # print('pn\n', raster_settings.projmatrix)
        # print('tfovx', raster_settings.tanfovx)
        # print('tfovy', raster_settings.tanfovy)
        # print(raster_settings.image_height)
        # print(raster_settings.image_width)
        # print(raster_settings.prefiltered)


        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = rasterize_gaussians_filter(means3D,
                                                 scales,
                                                 rotations,
                                                 raster_settings.scale_modifier,
                                                 cov3D_precomp,
                                                 raster_settings.viewmatrix.flatten(),
                                                 raster_settings.projmatrix.flatten(),
                                                 raster_settings.tanfovx,
                                                 raster_settings.tanfovy,
                                                 raster_settings.image_height,
                                                 raster_settings.image_width,
                                                 raster_settings.prefiltered,
                                                 raster_settings.debug)
        return radii