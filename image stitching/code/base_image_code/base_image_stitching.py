import os
import cv2 as cv
import numpy as np
from base_Video_preprocessing import video_stitching, anorm2, anorm, matchKeypoints, drawMatches

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

        # 이미지 저장
        image_name = f'base_black_cut_{i}.png'
        cv.imwrite(image_name, images[i])
        print(f"base_black_cut_image {i} as {image_name}")

    # 검은 부분이 제거된 이미지 리스트를 선택된 이미지 리스트에 추가
    selected_images = images

    return selected_images

def read_image_list_from_file(file_path):
    with open(file_path, 'r') as file:
        image_list = [line.strip() for line in file]

    return image_list

def load_images_from_list(image_list):
    images = [cv.imread(image_name) for image_name in image_list]
    return images

def main():
    global selected_images
    i = 0
    # 현재 스크립트 파일이 위치한 디렉토리, 이미지 리스트 가져옴
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # 이미지 리스트 파일에서 이미지 이름을 읽어옴
    image_list = read_image_list_from_file('image_list(base).txt')

    # 이미지를 읽어와서 리스트에 추가
    loaded_images = load_images_from_list(image_list)

    # 후 처리를 위해 이미지 선택
    selected_images = post_process_images(loaded_images)

    # 첫 번째 이미지 읽어오기
    first_image_path = image_list[i]
    prev_image = cv.imread(first_image_path)

    # 그레이스케일로 변환
    prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)

    # 결과 이미지 초기화
    result = prev_image

    i += 1

    while i < len(image_list):
        # 다음 프레임 읽기
        image_path = image_list[i]
        current_image = cv.imread(image_path)

        # 현재 프레임을 그레이스케일로 변환
        current_gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

        # BRISK 디텍터로 키포인트 및 디스크립터 추출
        detector = cv.BRISK_create()
        keyPoints1, descriptors1 = detector.detectAndCompute(prev_gray, None)
        keyPoints2, descriptors2 = detector.detectAndCompute(current_gray, None)

        # 키포인트 좌표를 부동소수점 형식으로 변환
        keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
        keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])

        # 키포인트 매칭 및 호모그래피 추정
        matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2)

        # 매칭 결과 이미지 생성
        img_matching_result = drawMatches(prev_image, current_image, keyPoints1, keyPoints2, matches, status)

        # 이미지 스티칭 부분 수정
        try:
            # 호모그래피 계산
            H, _ = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC, 5.0)

            # 현재 프레임 전체를 이전 결과 이미지에 추가
            result = cv.warpPerspective(current_image, H, (result.shape[1] + current_image.shape[1], result.shape[0]))

        except cv.error as e:
            # 이미지 크기 제한 초과 시 오류 처리
            print(f"에러: {e}")
            print("스티칭 재시작.")
            # 결과 이미지 초기화
            result = current_image

        # 현재 프레임을 이전 프레임으로 설정
        prev_image = current_image
        prev_gray = current_gray

        # 결과 이미지 표시
        cv.imshow('result', result)
        cv.imshow('matching result', img_matching_result)
        cv.imwrite('result_image.jpg', result)
        cv.imwrite('matching_result_image.jpg', img_matching_result)

        # 종료 키 (예: ESC 키) 입력을 기다림
        if cv.waitKey(30) & 0xFF == 27:
            break
        i += 1
    cv.destroyAllWindows()

def image_list_check(file_path):
    global loaded_image_list
    global image_list

    try:
        with open(file_path, 'r') as file:
            loaded_image_list = file.readlines()
        # 파일이 비어있는지 확인
        if not loaded_image_list:
            print("파일은 존재하지만 비어있습니다.")
            video_stitching()

        else:
            print("파일 내용:", loaded_image_list)
            return loaded_image_list

    except FileNotFoundError:
        print(f"파일이 존재하지 않습니다: {file_path}")
        video_stitching()

    except Exception as e:
        print(f"다른 예외가 발생했습니다: {e}")
        return None

if __name__ == '__main__':
    image_list_check('image_list.txt')
    main()
