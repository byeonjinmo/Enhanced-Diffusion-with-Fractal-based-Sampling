"""

Generate a large batch of image samples from a model and save them as a large

numpy array. This can be used to produce samples for FID evaluation.

"""



import argparse

import os

import time

import numpy as np

import torch as th

import torch.distributed as dist



from improved_diffusion import dist_util, logger

from improved_diffusion.script_util import (

    NUM_CLASSES,

    model_and_diffusion_defaults,

    create_model_and_diffusion,

    add_dict_to_argparser,

    args_to_dict,

)

import cv2
import numpy as np


#def compute_laplacian_pyramid(image, levels):
 #   gaussian_pyramid = [image]
  #  for i in range(levels):
   #     image = cv2.pyrDown(image)
    #    gaussian_pyramid.append(image)

    #laplacian_pyramid = []
    #for i in range(levels, 0, -1):
     #   gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
      #  laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
       # laplacian_pyramid.append(laplacian)

    #return laplacian_pyramid
import torch
import torch.nn.functional as F

def gaussian_pyramid(image, num_levels):
    gaussian_images = [image]
    for level in range(1, num_levels):
        image = F.avg_pool2d(image, kernel_size=2, stride=2)
        gaussian_images.append(image)
    return gaussian_images

def laplacian_pyramid(gaussian_images):
    laplacian_images = []
    num_levels = len(gaussian_images)
    for level in range(num_levels - 1):
        upsampled = F.interpolate(gaussian_images[level + 1], scale_factor=2, mode='nearest')
        laplacian = gaussian_images[level] - upsampled
        laplacian_images.append(laplacian)
    # The last level of the Gaussian pyramid is also the last level of the Laplacian pyramid
    laplacian_images.append(gaussian_images[-1])
    return laplacian_images


def calculate_fractal_mu(mu_0, image, num_levels, weights):
    gaussian_images = gaussian_pyramid(image, num_levels)
    laplacian_images = laplacian_pyramid(gaussian_images)

    mu_fractal = mu_0
    for k, laplacian in enumerate(laplacian_images):
        mu_fractal += weights[k] * laplacian

    return mu_fractal


def main():

    args = create_argparser().parse_args()



    dist_util.setup_dist()

    logger.configure()



    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(

        **args_to_dict(args, model_and_diffusion_defaults().keys())

    )

    model.load_state_dict(

        dist_util.load_state_dict(args.model_path, map_location="cpu")

    )

    model.to(dist_util.dev())

    model.eval()



    logger.log("sampling...")

    all_images = []

    all_labels = []

    start_time = time.time()  # 샘플링 시작 시간 기록

    while len(all_images) * args.batch_size < args.num_samples:

        model_kwargs = {}

        if args.class_cond:

            classes = th.randint(

                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()

            )

            model_kwargs["y"] = classes

        sample_fn = (

            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop

        )

        sample = sample_fn(

            model,

            (args.batch_size, 3, args.image_size, args.image_size),

            clip_denoised=args.clip_denoised,

            model_kwargs=model_kwargs,

        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)

        sample = sample.permute(0, 2, 3, 1)

        sample = sample.contiguous()



        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]

        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL

        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        if args.class_cond:

            gathered_labels = [

                th.zeros_like(classes) for _ in range(dist.get_world_size())

            ]

            dist.all_gather(gathered_labels, classes)

            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        logger.log(f"created {len(all_images) * args.batch_size} samples")

        # 새로 추가할 샘플링 진행 상황 로깅

        num_created_samples = len(all_images) * args.batch_size

        num_remaining_samples = args.num_samples - num_created_samples

        num_remaining_batches = np.ceil(num_remaining_samples / args.batch_size)

        elapsed_time = time.time() - start_time

        estimated_total_time = elapsed_time / num_created_samples * args.num_samples

        estimated_remaining_time = estimated_total_time - elapsed_time

        logger.log(

            f"Created {num_created_samples} samples, Remaining batches: {num_remaining_batches}, Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {estimated_remaining_time:.2f}s")



    arr = np.concatenate(all_images, axis=0)

    arr = arr[: args.num_samples]

    if args.class_cond:

        label_arr = np.concatenate(all_labels, axis=0)

        label_arr = label_arr[: args.num_samples]

    if dist.get_rank() == 0:

        shape_str = "x".join([str(x) for x in arr.shape])

        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")

        logger.log(f"saving to {out_path}")

        if args.class_cond:

            np.savez(out_path, arr, label_arr)

        else:

            np.savez(out_path, arr)



    dist.barrier()

    logger.log("sampling complete")





def create_argparser():

    defaults = dict(

        clip_denoised=True,

        num_samples=1000,  # 10000

        batch_size=32,    # 16

        use_ddim=False,

        model_path="",

    )

    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()

    add_dict_to_argparser(parser, defaults)

    return parser





if __name__ == "__main__":

    main()