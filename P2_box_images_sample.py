import argparse
import os
import time
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

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
    laplacian_images.append(gaussian_images[-1])
    return laplacian_images

def calculate_weights(num_levels, fractal_dimension):
    weights = [1 / (k ** fractal_dimension) for k in range(1, num_levels + 1)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return normalized_weights

# 프랙탈 차원 계산 함수 (box_counting) 추가
# def box_counting(image, min_box_size=1, max_box_size=None, steps=10):
#     if max_box_size is None:
#         max_box_size = min(image.size(2), image.size(3)) // 2
#     box_sizes = np.floor(np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=steps)).astype(int)
#     counts = []
#     for size in box_sizes:
#         stride = size
#         num_boxes = 0
#         for y in range(0, image.size(2) - size + 1, stride):
#             for x in range(0, image.size(3) - size + 1, stride):
#                 region = image[:, :, y:y+size, x:x+size]
#                 if th.sum(region) > 0:
#                     num_boxes += 1
#         counts.append(num_boxes)
#     coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
#     fractal_dimension = coeffs[0]
#     return fractal_dimension
def box_counting(image, min_box_size=1, max_box_size=None, steps=10):
    # 이미지를 흑백으로 변환하고, numpy 배열로 변환
    image = image.mean(dim=0).numpy()
    if max_box_size is None:
        max_box_size = min(image.shape) // 2
    sizes = np.floor(np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=steps)).astype(int)
    counts = []
    for size in sizes:
        count = 0
        for x in range(0, image.shape[1] - size + 1, size):
            for y in range(0, image.shape[0] - size + 1, size):
                if np.sum(image[y:y+size, x:x+size]) > 0:
                    count += 1
        counts.append(count)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dimension = coeffs[0]
    return fractal_dimension


# 프랙탈 차원에 기반한 그래디언트 조정 계산 함수
def calculate_fractal_adjustment(fractal_dim):
    adjustment_factor = 0.1
    gradient_adjustment = fractal_dim * adjustment_factor
    return gradient_adjustment

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

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
    start_time = time.time()
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

        # 여기에 프랙탈 차원에 기반한 가중치 적용 로직 추가
        # 프랙탈 차원 계산 및 적용 예시는 생략됨

        # 이미지 후처리 및 저장 코드는 변함 없음
        # ...
        # 프랙탈 기반 가중치 계산과 라플라시안 피라미드 적용 부분
        num_levels = 5
        # 예시로 5단계의 라플라시안 피라미드 사용

        for i in range(sample.shape[0]):
            image = sample[i].unsqueeze(0)  # 배치 차원을 유지하면서 i번째 이미지 선택
            # 프랙탈 차원 계산
            fractal_dimension = box_counting(image)
            # 프랙탈 차원을 기반으로 가중치 계산
            weights = calculate_weights(num_levels, fractal_dimension)

            gaussian_images = gaussian_pyramid(image, num_levels)
            laplacian_images = laplacian_pyramid(gaussian_images)
            mu_fractal = image  # 초기 이미지
            for k, laplacian in enumerate(laplacian_images):
                _, _, H, W = sample.shape
                upsampled_laplacian = F.interpolate(laplacian, size=(H, W), mode='bilinear', align_corners=False)
                mu_fractal += weights[k] * upsampled_laplacian
            sample[i] = mu_fractal.squeeze(0)  # 다시 배치에 삽입
        # 이미지 후처리 및 저장 코드
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # NCCL에서는 gather 지원 안 함
        all_images.extend([s.cpu().numpy() for s in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([l.cpu().numpy() for l in gathered_labels])

        # 진행 상황 로깅
        num_created_samples = len(all_images) * args.batch_size
        num_remaining_samples = args.num_samples - num_created_samples
        num_remaining_batches = np.ceil(num_remaining_samples / args.batch_size)
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / num_created_samples * args.num_samples
        estimated_remaining_time = estimated_total_time - elapsed_time
        logger.log(
            f"Created {num_created_samples} samples, "
            f"Remaining batches: {num_remaining_batches}, "
            f"Elapsed time: {elapsed_time:.2f}s, "
            f"Estimated remaining time: {estimated_remaining_time:.2f}s"
        )

        # 이미지를 numpy 배열로 변환하고 저장
    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[:args.num_samples]
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

if __name__ == "__main__":
    main()
