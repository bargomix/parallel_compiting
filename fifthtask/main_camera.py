import argparse
import queue
import threading
import time

import cv2


def load_model():
    try:
        from ultralytics import YOLO
    except ImportError as err:
        raise RuntimeError("Установите ultralytics: python3 -m pip install ultralytics") from err

    return YOLO("yolov8s-pose.pt")


def parse_camera(value):
    if value.isdigit():
        return int(value)

    return value


def put_latest(q, value):
    while True:
        try:
            q.put_nowait(value)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def get_latest(q, previous):
    latest = previous

    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest


def worker(worker_id, input_queue, output_queue, stop_event):
    model = load_model()

    while not stop_event.is_set():
        try:
            frame_id, frame = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if frame is None:
            break

        try:
            results = model(frame, verbose=False)
            put_latest(output_queue, (frame_id, results[0].plot()))
        except Exception as err:
            print(f"Ошибка в воркере {worker_id}: {err}")
            put_latest(output_queue, (frame_id, frame))

    print(f"Воркер {worker_id} завершил работу")


def draw_fps(frame, fps):
    text = f"FPS: {fps:.1f}"
    cv2.rectangle(frame, (10, 10), (130, 45), (255, 255, 255), -1)
    cv2.rectangle(frame, (10, 10), (130, 45), (30, 30, 30), 1)
    cv2.putText(frame, text, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    return frame


def build_parser():
    parser = argparse.ArgumentParser(description="Realtime YOLO pose для камеры")
    parser.add_argument("--camera", default="0", help="Номер камеры или путь, например 0 или /dev/video0")
    parser.add_argument("--num_workers", type=int, default=2, help="Количество потоков")
    parser.add_argument("--width", type=int, default=640, help="Ширина кадра")
    parser.add_argument("--height", type=int, default=480, help="Высота кадра")
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_workers < 1:
        raise RuntimeError("Количество потоков должно быть положительным")

    cap = cv2.VideoCapture(parse_camera(args.camera))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру: {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    input_queue = queue.Queue(maxsize=args.num_workers)
    output_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    workers = []

    for i in range(args.num_workers):
        thread = threading.Thread(
            target=worker,
            args=(i, input_queue, output_queue, stop_event),
        )
        thread.start()
        workers.append(thread)

    frame_id = 0
    shown_frames = 0
    latest_frame = None
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            put_latest(input_queue, (frame_id, frame))
            frame_id += 1

            latest_result = get_latest(output_queue, None)
            if latest_result is not None:
                _, latest_frame = latest_result

            elapsed = time.time() - start_time
            fps = shown_frames / elapsed if elapsed > 0 else 0.0
            image = latest_frame.copy() if latest_frame is not None else frame

            cv2.imshow("Task5 realtime pose", draw_fps(image, fps))
            shown_frames += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        for _ in workers:
            put_latest(input_queue, (None, None))

        for thread in workers:
            thread.join()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
