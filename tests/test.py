#!/usr/bin/env python3
import cv2
import numpy as np

def write_pose(file_handle, t, R):
    """
    Записывает в файл позу в виде: tx ty tz r11 r21 r31 r12 r22 r32 r13 r23 r33
    где столбцы матрицы поворота записаны по столбцам.
    """
    # Приводим к виду одномерных массивов
    t = t.flatten()
    R = R.flatten('F')  # порядок по столбцам
    pose = np.hstack((t, R))
    line = ' '.join(f'{num:.6f}' for num in pose)
    file_handle.write(line + '\n')

def main():
    # Параметры видео и камеры (пример, замените на свои реальные параметры)
    video_path = "datasets/output.mp4"  # путь к синтетическому видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Не удалось открыть видео.")
        return

    # Пример параметров камеры: матрица K (замените на реальные значения)
    fx = 615
    fy = 615
    cx = 320
    cy = 240
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # Инициализация детектора особенностей
    orb = cv2.ORB_create(2000)

    # Чтение первого кадра
    ret, prev_frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр.")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # Детектируем особенности в первом кадре
    prev_keypoints = orb.detect(prev_gray, None)
    prev_keypoints, prev_des = orb.compute(prev_gray, prev_keypoints)
    prev_pts = np.array([kp.pt for kp in prev_keypoints], dtype=np.float32)

    # Начальные параметры движения: тождественная матрица
    R_total = np.eye(3)
    t_total = np.zeros((3,1))

    # Файл для записи траектории
    output_file = open('trajectory.txt', 'w')
    # Записываем первую позу (начальное положение)
    write_pose(output_file, t_total, R_total)

    # Параметры для оптического потока
    lk_params = dict(winSize=(21, 21),
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Отслеживаем точки из предыдущего кадра
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        # Выбираем те, которые корректно отследились
        good_prev_pts = prev_pts[status.flatten() == 1]
        good_next_pts = next_pts[status.flatten() == 1]

        # Если точек слишком мало, повторно детектируем особенности
        if len(good_prev_pts) < 50:
            keypoints = orb.detect(prev_gray, None)
            keypoints, des = orb.compute(prev_gray, keypoints)
            prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            good_prev_pts = prev_pts[status.flatten() == 1]
            good_next_pts = next_pts[status.flatten() == 1]

        # Вычисляем эссенциальную матрицу
        if len(good_prev_pts) < 5:
            print(f"Недостаточно точек на кадре {frame_idx}, пропускаем кадр.")
            prev_gray = gray.copy()
            prev_pts = good_next_pts.reshape(-1, 1, 2)
            frame_idx += 1
            continue

        E, mask = cv2.findEssentialMat(good_next_pts, good_prev_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print(f"Не удалось найти эссенциальную матрицу на кадре {frame_idx}.")
            prev_gray = gray.copy()
            prev_pts = good_next_pts.reshape(-1, 1, 2)
            frame_idx += 1
            continue

        # Восстанавливаем поворот и перевод
        _, R, t, mask_pose = cv2.recoverPose(E, good_next_pts, good_prev_pts, K)
        # Накопление движения
        t_total += R_total @ t
        R_total = R @ R_total

        # Записываем текущую позу в файл
        write_pose(output_file, t_total, R_total)

        # Подготовка к следующей итерации
        prev_gray = gray.copy()
        prev_pts = good_next_pts.reshape(-1, 1, 2)
        frame_idx += 1

    cap.release()
    output_file.close()
    print("Обработка завершена, траектория сохранена в 'trajectory.txt'.")

if __name__ == '__main__':
    main()
