
import argparse
import os
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


# FractalTransformNet 정의
class FractalTransformNet(nn.Module):
    def __init__(self):
        super(FractalTransformNet, self).__init__()
        # 컨볼루션 레이어 정의
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, fractal_dimension):
        # 컨볼루션 및 비선형 활성화 적용
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        # 프랙탈 차원으로 출력 스케일 조정
        return x * fractal_dimension


# 가우시안 및 라플라시안 피라미드 함수, 가중치 계산 함수 등

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
    laplacian_images.append(gaussian_images[-1])  # 마지막 Gaussian 이미지는 Laplacian 피라미드의 마지막 레벨도 됨
    return laplacian_images

def calculate_weights(num_levels, fractal_dimension):
    weights = [1 / (k ** fractal_dimension) for k in range(1, num_levels + 1)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return normalized_weights

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        clip_denoised=True,
        num_samples=10000,
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

    # FractalTransformNet 인스턴스 초기화
    fractal_net = FractalTransformNet().to(dist_util.dev())

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

    # 프랙탈 기반 가중치 계산 및 라플라시안 피라미드 변환 적용
    num_levels = 6
    fractal_dimension = 2.3  # 프랙탈 차원, 이는 조정 가능한 파라미터
    weights = calculate_weights(num_levels, fractal_dimension)

    for i in range(sample.shape[0]):
        image = sample[i].unsqueeze(0)  # 배치 차원을 유지하면서 i번째 이미지 선택
        gaussian_images = gaussian_pyramid(image, num_levels)
        laplacian_images = laplacian_pyramid(gaussian_images)
        mu_fractal = image  # mu_0 이미지 초기화
        for k, laplacian in enumerate(laplacian_images):
            _, _, H, W = sample.shape
            upsampled_laplacian = F.interpolate(laplacian, size=(H, W), mode='bilinear', align_corners=False)
            mu_fractal += weights[k] * upsampled_laplacian
        transformed_image = fractal_net(mu_fractal, fractal_dimension)
        sample[i] = transformed_image.squeeze(0)  # 배치에 다시 삽입

    # 이미지 후처리 및 저장
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)  # HWC 형식으로 변환하여 이미지로 저장
    sample = sample.contiguous()

    # 이미지 저장 또는 처리 로직
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
    logger.log("샘플링 완료")

if __name__ == "__main__":
    main()
