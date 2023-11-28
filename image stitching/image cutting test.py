import os
import cv2 as cv
import numpy as np

def post_process_images(images):
    global selected_images
    image_matrices = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    selected_images = []

    # 각 이미지에 대해 검은 부분 제거
    for i in range(len(image_matrices)):
        # 행 제거
        non_zero_rows = np.any(image_matrices[i] != 0, axis=1)
        images[i] = images[i][non_zero_rows, :]

        # 열 제거
        non_zero_cols = np.any(image_matrices[i] != 0, axis=0)
        images[i] = images[i][:, non_zero_cols]

    # 검은 부분이 제거된 이미지 리스트를 선택된 이미지 리스트에 추가
    selected_images.append(images)

    return selected_images

if __name__ == '__main__':
    # 예제 이미지 파일 경로
    image_path = 'result_105.0.png'

    # 이미지 불러오기
    image = cv.imread(image_path)

    # 이미지를 리스트로 변환하여 함수에 전달
    result_images = post_process_images([image])

    # 결과 출력
    for i, result_image in enumerate(result_images[0]):
        cv.imshow(f'Result Image {i + 1}', result_image)

    cv.waitKey(0)
    cv.destroyAllWindows()
