import cv2 as cv
def start_point(video_route):
    # 비디오 파일 열기
    cap = cv.VideoCapture(video_route)

    # 비디오가 성공적으로 열렸는지 확인
    if not cap.isOpened():
        print("에러: 비디오를 열 수 없습니다.")
        return

    # 첫 프레임 읽어오기
    ret, frame = cap.read()

    # 첫 프레임이 정상적으로 읽혔는지 확인
    if not ret:
        print("에러: 첫 프레임을 읽을 수 없습니다.")
        return

    # 첫 프레임을 PNG로 저장
    image_name = f'start_point.png'
    cv.imwrite(image_name, frame)
    # 비디오 캡처 객체 해제
    cap.release()

    print(f"첫 프레임이 저장 완료")

