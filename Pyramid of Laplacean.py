import cv2
import numpy as np

# 원본 이미지 불러오기
image = cv2.imread('/Users/mac/Desktop/cv_jin/diffusion_git_c/proposed/a_photograph_of_an_astronaut_riding_a_horse_S3202469941_St25_G7.5.jpeg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 가우시안 피라미드를 생성하기 위한 함수
def gaussian_pyramid(image, levels):
    lower = image.copy()
    gaussian_pyr = [lower]
    for i in range(levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(lower)
    return gaussian_pyr


# 라플라시안 피라미드를 생성하기 위한 함수
def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i])
        gaussian_upper = gaussian_pyr[i - 1]
        laplacian = cv2.subtract(gaussian_upper, gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


# 가우시안 및 라플라시안 피라미드 생성
num_levels = 5  # 레벨 수 정의
gaussian_pyr = gaussian_pyramid(image, num_levels)
laplacian_pyr = laplacian_pyramid(gaussian_pyr)

# 라플라시안 피라미드 이미지 저장
for i, lap in enumerate(laplacian_pyr):
    cv2.imwrite(f'laplacian_level_{i}.jpg', lap)


