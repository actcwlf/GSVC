import time

import torch

from common.base import RenderResults
from diff_gaussian_rasterization.cuda_ortho_gaussian_rasterizer import GaussianRasterizationSettings, GaussianRasterizer

from frame_cube.frame import Frame
from scene.gaussian_model import GaussianModel
from ortho_gaussian_renderer import generate_neural_gaussians, GenerateMode
from ortho_gaussian_renderer.preprocess import prefilter_voxel


def render(
        frame: Frame,
        pc: GaussianModel,
        pipe,
        bg_color : torch.Tensor,
        scaling_modifier = 1.0,
        retain_grad=False,
        mode=GenerateMode.TRAINING_FULL_PRECISION
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    visible_mask = prefilter_voxel(frame, pc, pipe, bg_color)


    gss = generate_neural_gaussians(frame, pc, visible_mask, mode)

    # min_x, max_x = xyz[:, 0].min(), xyz[:, 0].max()
    # min_y, max_y = xyz[:, 1].min(), xyz[:, 1].max()
    # min_z, max_z = xyz[:, 2].min(), xyz[:, 2].max()

    screenspace_points = torch.zeros_like(gss.xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    #
    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(viewpoint_camera.image_height),
    #     image_width=int(viewpoint_camera.image_width),
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=viewpoint_camera.world_view_transform,
    #     projmatrix=viewpoint_camera.full_proj_transform,
    #     sh_degree=1,
    #     campos=viewpoint_camera.camera_center,
    #     prefiltered=False,
    #     debug=pipe.debug
    # )

    raster_settings = GaussianRasterizationSettings(
        image_height=int(frame.image_height),
        image_width=int(frame.image_width),
        # tanfovx=tanfovx,
        # tanfovy=tanfovy,
        x_min=frame.x_min,
        y_min=frame.y_min,
        scale=frame.scale,
        threshold=pc.model_config.threshold,
        # kernel_size=pc.model_config.kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=frame.view_matrix.permute(1, 0).cuda(),
        # viewmatrix=frame.view_matrix.cuda(),
        sh_degree=pc.model_config.sh_degree,
        campos=frame.cam_pos,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)



    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, num_rendered = rasterizer(
        means3D = gss.xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = gss.color,
        opacities = gss.opacity,
        scales = gss.scaling,
        rotations = gss.rot,
        cov3D_precomp = None)


    return RenderResults(
        rendered_image=rendered_image,
        viewspace_points=screenspace_points,
        visibility_filter=radii > 0,
        visible_mask=visible_mask,
        radii=radii,
        active_gaussains=(radii > 0).sum(),
        num_rendered=num_rendered,
        selection_mask=gss.mask,
        neural_opacity=gss.neural_opacity,
        scaling=gss.scaling,
        bit_per_param=gss.bit_per_param,
        bit_per_feat_param=gss.bit_per_feat_param,
        bit_per_scaling_param=gss.bit_per_scaling_param,
        bit_per_offsets_param=gss.bit_per_offsets_param,
        entropy_constrained=(gss.bit_per_param is not None),
        generated_gaussians=gss,
        time_sub=gss.time_sub
    )


