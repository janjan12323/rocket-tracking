import cv2
import numpy as np
from start_point import start_point

def draw_arrow(img, p1, p2):
    # Draw an arrow on the image from point p1 to point p2
    cv2.arrowedLine(img, tuple(p1), tuple(p2), (0, 0, 255), 2)


def main():
    # 베이스 이미지 로드
    base_image = cv2.imread('base_image.png')

    # 루트 이미지 로드
    route_image = cv2.imread('route_image.png')

    # 시작점 이미지 로드
    start_point_image = cv2.imread('start_point.png')

    # 이미지 1과 이미지 2 간의 특징점 매칭
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(base_image, None)
    kp2, des2 = orb.detectAndCompute(route_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    # 이미지 1과 이미지 3 간의 특징점 매칭
    kp1, des1 = orb.detectAndCompute(base_image, None)
    kp3, des3 = orb.detectAndCompute(start_point_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des3)

    matches = sorted(matches, key=lambda x: x.distance)

    # 특징점 중심 좌표 계산
    center1 = np.array([sum([kp.pt[0] for kp in kp1]) / len(kp1),
                        sum([kp.pt[1] for kp in kp1]) / len(kp1)])

    center2 = np.array([sum([kp.pt[0] for kp in kp2]) / len(kp2),
                        sum([kp.pt[1] for kp in kp2]) / len(kp2)])

    center3 = np.array([sum([kp.pt[0] for kp in kp3]) / len(kp3),
                        sum([kp.pt[1] for kp in kp3]) / len(kp3)])

    # 이동 전후 정사각형 중심 및 이동 벡터 계산
    square_size = 100  # 정사각형의 크기

    while True:
        # 정사각형 그리기
        square = np.array([[center1[0] - square_size / 2, center1[1] - square_size / 2],
                           [center1[0] + square_size / 2, center1[1] - square_size / 2],
                           [center1[0] + square_size / 2, center1[1] + square_size / 2],
                           [center1[0] - square_size / 2, center1[1] + square_size / 2]])

        # 이미지 1에 화살표 그리기
        route_image = base_image.copy()
        draw_arrow(route_image, center1, center1 + (center3 - center2))

        # 이미지 표시
        cv2.imshow('Image 1 with Arrow', route_image)

        # 키 이벤트 처리
        key = cv2.waitKey(0)

        # ESC 키를 누르면 종료
        if key == 27:
            break

        # 화살표에 따라 정사각형 이동
        if key == ord('w'):
            center1[1] -= 5
        elif key == ord('a'):
            center1[0] -= 5
        elif key == ord('s'):
            center1[1] += 5
        elif key == ord('d'):
            center1[0] += 5

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_point(r'C:\Users\jangj\PycharmProjects\pythonProject1\image stitching\route video.mp4')
    main()
