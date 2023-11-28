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


def main():
    img1 = cv.imread('preprocessing27.0.png')
    img2 = cv.imread('preprocessing53.0.png')

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    detector = cv.BRISK_create()
    keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    print('img1 - %d features, img2 - %d features' % (len(keyPoints1), len(keyPoints2)))

    keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
    keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])

    matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2)

    img_matching_result = drawMatches(img1, img2, keyPoints1, keyPoints2, matches, status)

    result = cv.warpPerspective(img1, H,(img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    cv.imshow('result', result)
    cv.imshow('matching result', img_matching_result)

    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

