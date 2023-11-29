import os
import numpy as np
import cv2 as cv

FLANN_INDEX_LSH = 6


def anorm2(a):
    return (a * a).sum(-1)


def anorm(a):
    return np.sqrt(anorm2(a))

def matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2):
    flann_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)  # 2

    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.79:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) >= 4:

        keyPoints1 = np.float32([keyPoints1[i] for (_, i) in matches])
        keyPoints2 = np.float32([keyPoints2[i] for (i, _) in matches])

        H, status = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC, 4.0)

        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    return matches, H, status


def drawMatches(image1, image2, keyPoints1, keyPoints2, matches, status):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    img_matching_result = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")

    img_matching_result[0:h2, 0:w2] = image2
    img_matching_result[0:h1, w2:] = image1

    for ((trainIdx, queryIdx), s) in zip(matches, status):

        if s == 1:
            keyPoint2 = (int(keyPoints2[trainIdx][0]), int(keyPoints2[trainIdx][1]))
            keyPoint1 = (int(keyPoints1[queryIdx][0]) + w2, int(keyPoints1[queryIdx][1]))
            cv.line(img_matching_result, keyPoint1, keyPoint2, (0, 255, 0), 1)

    return img_matching_result


def video_stitching():
    global image_list
    image_list = []
    # 비디오 파일 열기
    cap = cv.VideoCapture(r'C:\Users\jangj\PycharmProjects\pythonProject1\image stitching\base video.mp4')

    # 첫 번째 프레임 읽기
    ret, prev_frame = cap.read()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    # 결과 이미지 초기화
    result = prev_frame

    # 이미지 처리에 사용할 블록 크기 정의
    block_size = 500  # 예시로 500x500 크기의 블록 사용

    while True:
        # 다음 프레임 읽기
        ret, current_frame = cap.read()

        # 비디오의 끝에 도달하면 종료
        if not ret:
            break

        # 현재 프레임을 그레이스케일로 변환
        current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

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
        img_matching_result = drawMatches(prev_frame, current_frame, keyPoints1, keyPoints2, matches, status)

        try:
            # 이미지 스티칭
            result = cv.warpPerspective(result, H, (result.shape[1] + current_frame.shape[1], result.shape[0]))

            # 현재 프레임 일부를 처리하고 이미지에 추가
            for x in range(0, current_frame.shape[1], block_size):
                for y in range(0, current_frame.shape[0], block_size):
                    # 현재 프레임의 남은 부분 크기 계산
                    block_width = min(block_size, current_frame.shape[1] - x)
                    block_height = min(block_size, current_frame.shape[0] - y)

                    block = current_frame[y:y + block_height, x:x + block_width]
                    result[y:y + block_height, x:x + block_width] = block

        except cv.error as e:
            # 이미지 크기 제한 초과 시 오류 처리
            print(f"Error: {e}")
            print("Saving current result and restarting stitching.")

            # 이미지 파일의 이름 기록
            image_name = f'preprocessing(base){cap.get(cv.CAP_PROP_POS_FRAMES)}.png'
            image_list.append(image_name)

            # 현재 결과 이미지 저장
            cv.imwrite(image_name, result)

            # 결과 이미지 초기화
            result = current_frame

        # 현재 프레임을 이전 프레임으로 설정
        prev_frame = current_frame
        prev_gray = current_gray

        # 결과 이미지 표시
        cv.imshow('result', result)
        cv.imshow('matching result', img_matching_result)

        # 종료 키 (예: ESC 키) 입력을 기다림
        if cv.waitKey(30) & 0xFF == 27:
            break

    # 비디오 파일 닫기
    cap.release()
    cv.destroyAllWindows()

    # 이미지 파일 이름 기록
    with open('image_list(base).txt', 'w') as file:
        for item in image_list:
            file.write("%s\n" % item)


if __name__ == '__main__':
    video_stitching()
