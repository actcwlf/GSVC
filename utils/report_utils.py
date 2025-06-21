import os
import pathlib
import time
from loguru import logger
import torch
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from torchvision.transforms import transforms
from tqdm import tqdm

from frame_cube.frame_cube import FrameCube
from lpipsPyTorch import lpips
from ortho_gaussian_renderer import GenerateMode
from utils.image_utils import psnr
from utils.loss_utils import ssim_func
from utils.loss_utils import l1_loss_func
from utils.metric_utils import msssim_fn, lpips_fn


def log_training_results(
        tb_writer,
        dataset_name,
        psnr_val,
        iteration,
        l1_loss,
        loss,
        anchor_num,
        active_gaussians,
        num_rendered,
        mask_ratio,
        elapsed
):
    order = 0
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_loss_patches/PSNR', psnr_val.item(), iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_loss_patches/l1_loss', l1_loss.item(), iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_loss_patches/total_loss', loss.item(), iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/iter_time', elapsed, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_loss_patches/anchor_num', anchor_num, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_loss_patches/num_rendered', num_rendered, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_loss_patches/active_gaussians', active_gaussians, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_loss_patches/mask_ratio', mask_ratio, iteration)


def training_report(
        tb_writer,
        dataset_name,
        psnr_val,
        iteration,
        Ll1,
        loss,
        l1_loss_func,
        elapsed,
        testing_iterations,
        frame_cube: FrameCube,
        renderFunc,
        renderArgs,
        wandb=None,
        logger=None,
        pre_path_name='',
        run_codec=True
):
    assert tb_writer

    # tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/PSNR', psnr_val.item(), iteration)
    # tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
    # tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
    # tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    # if wandb is not None:
    #     wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })


    # Report test and samples of training set
    if iteration not in testing_iterations:
        return
    frame_cube.gaussians.eval()

    # 在最后一个测试轮次执行编码
    if iteration == testing_iterations[-1]:
        with torch.no_grad():
            log_info, _ = frame_cube.gaussians.estimate_final_bits()
            logger.info(log_info)
        if run_codec:  # conduct encoding and decoding
            with torch.no_grad():
                bit_stream_path = os.path.join(pre_path_name, 'bitstreams')
                os.makedirs(bit_stream_path, exist_ok=True)
                # conduct encoding
                patched_infos, log_info, _ = frame_cube.gaussians.conduct_encoding(pre_path_name=bit_stream_path)
                logger.info(log_info)
                # conduct decoding
                log_info = frame_cube.gaussians.conduct_decoding(pre_path_name=bit_stream_path, patched_infos=patched_infos)
                logger.info(log_info)
    torch.cuda.empty_cache()

    config = {
        'name': 'test',
        'frames': frame_cube.dataset.len_z_frames
    }

    l1_test = 0.0
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0

    if wandb is not None:
        gt_image_list = []
        render_image_list = []
        errormap_list = []

    t_list = []

    # for idx, viewpoint in enumerate(config['cameras']):
    # for idx, frame in enumerate(frame_cube.dataset[:frame_cube.dataset.len_z_frames]):
    for idx in range(frame_cube.dataset.len_z_frames):
        frame = frame_cube.dataset[idx]
        torch.cuda.synchronize(); t_start = time.time()
        # voxel_visible_mask = prefilter_voxel(frame, frame_cube.gaussians, *renderArgs)
        # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
        render_results1 = renderFunc(frame, frame_cube.gaussians, *renderArgs)
        frame.view_matrix = frame.view_matrix_s.cuda()
        render_results2 = renderFunc(frame, frame_cube.gaussians, *renderArgs)
        image1 = render_results1.rendered_image
        image2 = render_results2.rendered_image
        image2_f = torch.flip(image2, dims=(-1,))

        image = (image1 + image2_f) / 2


        image = torch.clamp(image, 0.0, 1.0)
        time_sub = render_results1.time_sub + render_results2.time_sub
        torch.cuda.synchronize(); t_end = time.time()
        t_list.append(t_end - t_start - time_sub)

        gt_image = torch.clamp(frame.image.to("cuda"), 0.0, 1.0).permute(0, 2, 1)
        if tb_writer and (idx < 30):
            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(frame.image_id), image[None], global_step=iteration)
            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(frame.image_id), (gt_image[None]-image[None]).abs(), global_step=iteration)

            # if wandb:
            #     render_image_list.append(image[None])
            #     errormap_list.append((gt_image[None]-image[None]).abs())

            if iteration == testing_iterations[0]:
                tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(frame.image_id), gt_image[None], global_step=iteration)
                # if wandb:
                #     gt_image_list.append(gt_image[None])
        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        ssim_test += ssim_func(image, gt_image).mean().double()
        # lpips_test += lpips_fn(image, gt_image, normalize=True).detach().mean().double()
        # lpips_test += lpips(image, gt_image, net_type='vgg').detach().mean().double()

    psnr_test /=  config['frames']
    ssim_test /=  config['frames']
    lpips_test /= config['frames']
    l1_test /=    config['frames']
    logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} lpips {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
    test_fps = 1.0 / torch.tensor(t_list[0:]).mean()
    logger.info(f'Test FPS: {test_fps.item():.5f}')
    # if tb_writer:

    tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
    # if wandb is not None:
    #     wandb.log({"test_fps": test_fps, })

    # if tb_writer:
    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
    # if wandb is not None:
    #     wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test}, f"ssim{ssim_test}", f"lpips{lpips_test}")

    # if tb_writer:
    # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

    tb_writer.add_scalar(f'{dataset_name}/'+'total_points', frame_cube.gaussians.get_anchor.shape[0], iteration)

    torch.cuda.empty_cache()

    frame_cube.gaussians.train()



def show_image(img: torch.Tensor, title):
    plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
    plt.title(title)
    plt.show()



@torch.no_grad()
def evaluate_one_frame(
        tb_writer,
        dataset_name,
        frame_cube: FrameCube,
        renderFunc,
        renderArgs,
        iteration=0,
        idx=0

):
    l1_test = 0.0
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    t_list = []

    # iteration = 0

    frame_cube.gaussians.eval()

    # idx = 0
    frame = frame_cube.dataset[idx]
    torch.cuda.synchronize();
    t_start = time.time()
    # voxel_visible_mask = prefilter_voxel(frame, frame_cube.gaussians, *renderArgs)
    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
    render_results1 = renderFunc(frame, frame_cube.gaussians, *renderArgs, mode=GenerateMode.TRAININ_STE_ENTROPY)
    frame.view_matrix = frame.view_matrix_s.cuda()
    render_results2 = renderFunc(frame, frame_cube.gaussians, *renderArgs, mode=GenerateMode.TRAININ_STE_ENTROPY)
    image1 = render_results1.rendered_image
    image2 = render_results2.rendered_image
    image2_f = torch.flip(image2, dims=(-1,))

    # img_input = torch.cat([image1, image2_f], dim=0).unsqueeze(0)
    # image = frame_cube.gaussians.conv_super_res(img_input).squeeze(0)
    #
    image = (image1 + image2_f) / 2

    # if idx % 100 == 0:
    #     show_image(image, f'idx {idx}')

    image = torch.clamp(image, 0.0, 1.0)
    # time_sub = render_results1.time_sub + render_results2.time_sub
    torch.cuda.synchronize();
    t_end = time.time()
    # t_list.append(t_end - t_start - time_sub)

    gt_image = torch.clamp(frame.image.to("cuda"), 0.0, 1.0).permute(0, 2, 1)
    # if tb_writer and (idx < 30):
    #     tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(frame.image_id), image[None], global_step=iteration)
    #     tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(frame.image_id), (gt_image[None]-image[None]).abs(), global_step=iteration)

    # if wandb:
    #     render_image_list.append(image[None])
    #     errormap_list.append((gt_image[None]-image[None]).abs())

    # if iteration == testing_iterations[0]:
    #     tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(frame.image_id), gt_image[None], global_step=iteration)
    # if wandb:
    #     gt_image_list.append(gt_image[None])
    l1_test += l1_loss_func(image, gt_image).mean().double().item()
    psnr_test += psnr(image, gt_image).mean().double().item()
    ssim_test += ssim_func(image, gt_image).mean().double().item()
    if iteration != 0:

        if iteration % 100 == 1:
            show_image(image, f'one frame idx {idx}')
        tb_writer.add_scalar(f'{dataset_name}/0/eval_first/PSNR', psnr_test, iteration)
    else:
        logger.info(f'First only test: L1 {l1_test} PSNR {psnr_test} SSIM {ssim_test}')
    frame_cube.gaussians.train()



@torch.no_grad()
def evaluate(
        tb_writer,
        dataset_name,
        frame_cube: FrameCube,
        renderFunc,
        # renderArgs,
        iteration = 0,
        order=3
):

    l1_test = 0.0
    psnr_test = 0.0
    ssim_test = 0.0
    msssim_test = 0.0
    lpips_test = 0.0
    t_list = []



    config = {
        'name': 'evaluation',
        'frames': frame_cube.dataset.len_z_frames
    }
    frame_cube.gaussians.eval()

    # warmup cuda
    frame = frame_cube.dataset[0]
    render_results1 = renderFunc(frame, frame_cube.gaussians)

    for idx in tqdm(range(frame_cube.dataset.len_z_frames), disable=True):
        frame = frame_cube.dataset[idx]
        torch.cuda.synchronize()
        t_start = time.time()
        # voxel_visible_mask = prefilter_voxel(frame, frame_cube.gaussians, *renderArgs)
        # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
        render_results1 = renderFunc(frame, frame_cube.gaussians)
        frame.view_matrix = frame.view_matrix_s.cuda()
        render_results2 = renderFunc(frame, frame_cube.gaussians)
        image1 = render_results1.rendered_image
        image2 = render_results2.rendered_image
        image2_f = torch.flip(image2, dims=(-1,))

        # img_input = torch.cat([image1, image2_f], dim=0).unsqueeze(0)
        # image = frame_cube.gaussians.conv_super_res(img_input).squeeze(0)
        #
        image = (image1 + image2_f) / 2
        image = torch.clamp(image, min=0,  max= 1.0)

        torch.cuda.synchronize()
        t_end = time.time()

        t_list.append(t_end - t_start)




        if idx % 100 == 0:
            show_image(image, f'idx {idx}')
        # time_sub = render_results1.time_sub + render_results2.time_sub
        # torch.cuda.synchronize(); t_end = time.time()
        # t_list.append(t_end - t_start - time_sub)

        gt_image = torch.clamp(frame.image.to("cuda"), 0, 1.0).permute(0, 2, 1)
        # if tb_writer and (idx < 30):
        #     tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(frame.image_id), image[None], global_step=iteration)
        #     tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(frame.image_id), (gt_image[None]-image[None]).abs(), global_step=iteration)

            # if wandb:
            #     render_image_list.append(image[None])
            #     errormap_list.append((gt_image[None]-image[None]).abs())

            # if iteration == testing_iterations[0]:
            #     tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(frame.image_id), gt_image[None], global_step=iteration)
                # if wandb:
                #     gt_image_list.append(gt_image[None])

        cur_psnr = peak_signal_noise_ratio(image.cpu().numpy(), gt_image.cpu().numpy(), data_range=1)
        cur_psnr2 = psnr(image, gt_image).mean().double().item()
        l1_test += l1_loss_func(image, gt_image).mean().double().item()
        psnr_test += cur_psnr # psnr(image, gt_image).mean().double().item()
        ssim_test += ssim_func(image, gt_image).mean().double().item()

        # print(f'idx {idx} PSNR', cur_psnr, cur_psnr2)
        # lpips_test += lpips_fn(image, gt_image, normalize=True).detach().mean().double()
        # lpips_test += lpips(image, gt_image, net_type='vgg').detach().mean().double()

        msssim_test += msssim_fn(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
        lpips_test += lpips_fn(image, gt_image, normalize=True).item()

        # break



    psnr_test /= config['frames']
    ssim_test /= config['frames']
    # lpips_test /= config['frames']
    l1_test /= config['frames']
    msssim_test /= config['frames']
    lpips_test /= config['frames']

    # if tb_writer:
    if order == 0: # runtime evaluation
        tb_writer.add_scalar(f'{dataset_name}/0/train/eval/PSNR', psnr_test, iteration)

        return

    # final evaluation
    logger.info(
        "[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} lpips {}".format(iteration, config['name'], l1_test,
                                                                         psnr_test, ssim_test, lpips_test))
    logger.info(
        "\n[ITER {}] Evaluating {}: L1 {} PSNR {} msssim {} lpips {}".format(iteration, config['name'],
                                                                           l1_test, psnr_test,
                                                                           msssim_test, lpips_test))




    # print(t_list)
    test_fps = 1.0 / torch.tensor(t_list).mean()
    logger.info(f'Test FPS: {test_fps.item():.5f}')

    tb_writer.add_scalar(f'{dataset_name}/{order}/eval/test_FPS', test_fps.item(), 0)
    # if wandb is not None:
    #     wandb.log({"test_fps": test_fps, })

    # if tb_writer:
    tb_writer.add_scalar(f'{dataset_name}/{order}/eval/l1_loss', l1_test, 0)
    tb_writer.add_scalar(f'{dataset_name}/{order}/eval/psnr', psnr_test, 0)
    tb_writer.add_scalar(f'{dataset_name}/{order}/eval/ssim', ssim_test, 0)
    tb_writer.add_scalar(f'{dataset_name}/{order}/eval/lpips', lpips_test, 0)
    # if wandb is not None:
    #     wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test}, f"ssim{ssim_test}", f"lpips{lpips_test}")

    # if tb_writer:
    # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

    tb_writer.add_scalar(f'{dataset_name}/{order}/eval/total_points', frame_cube.gaussians.get_anchor.shape[0], 0)

    torch.cuda.empty_cache()




@torch.no_grad()
def render_frames(
        # tb_writer,
        # dataset_name,
        frame_cube: FrameCube,
        renderFunc,
        output_dir: pathlib.Path
        # # renderArgs,
        # iteration = 0,
        # order=3
):

    frame_cube.gaussians.eval()

    to_image = transforms.ToPILImage()
    for idx in tqdm(range(frame_cube.dataset.len_z_frames)):
        frame = frame_cube.dataset.get_dummy_frame(idx)
        render_results1 = renderFunc(frame, frame_cube.gaussians)
        frame.view_matrix = frame.view_matrix_s.cuda()
        render_results2 = renderFunc(frame, frame_cube.gaussians)
        image1 = render_results1.rendered_image
        image2 = render_results2.rendered_image
        image2_f = torch.flip(image2, dims=(-1,))

        # img_input = torch.cat([image1, image2_f], dim=0).unsqueeze(0)
        # image = frame_cube.gaussians.conv_super_res(img_input).squeeze(0)
        #
        image = (image1 + image2_f) / 2
        image = torch.clamp(image, min=0,  max= 1.0)



        img = to_image(image)
        img.save(output_dir / f'd{idx:05d}.png')

        # show_image(image, f'idx {idx}')




