import argparse
import csv
import queue
import threading
import time
from pathlib import Path

import cv2


BENCHMARK_WORKERS = [2, 4, 7, 8, 16, 20, 40]


def load_model():
    try:
        from ultralytics import YOLO
    except ImportError as err:
        raise RuntimeError("Установите ultralytics: python3 -m pip install ultralytics") from err

    return YOLO("yolov8s-pose.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Многопоточная обработка кадров")
    parser.add_argument("path", help="путь к видео")
    parser.add_argument(
        "workmode", choices=["single", "multi", "benchmark"], help="Однопоточный/многопоточный/замеры"
    )
    parser.add_argument("name", help="Имя выходного файла")
    parser.add_argument("--num_workers", type=int, default=4, help="Количество потоков")
    return parser.parse_args()


def make_output_name(output_path, suffix):
    path = Path(output_path)
    ext = path.suffix or ".mp4"
    return str(path.with_name(f"{path.stem}_{suffix}{ext}"))


def make_csv_name(output_path):
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_timing.csv"))


class VideoProcessor:
    def __init__(self, video_path, output_path, num_workers=4):
        self.video_path = video_path
        self.output_path = output_path
        self.num_workers = num_workers

        # Буферы
        self.input_queue = queue.Queue(maxsize=50)
        self.output_queue = queue.Queue()
        self.ready_queue = queue.Queue()

        # Для восстановления порядка
        self.frame_buffer = {}
        self.next_frame_id = 0
        self.total_frames = 0

    def get_video_info(self):
        """Получить информацию о видео"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.fps <= 0:
            self.fps = 30.0

        if self.width <= 0 or self.height <= 0:
            raise RuntimeError("Не удалось получить размер видео")
        print(
            f"Видео: {self.width}x{self.height}, {self.fps:.2f} fps, {self.total_frames} кадров"
        )

    def read_video(self):
        """Чтение кадров из видео в входной буфер"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.video_path}")

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.input_queue.put((frame_id, frame))
            frame_id += 1

            if frame_id % 30 == 0:
                print(f"Прочитано {frame_id}/{self.total_frames} кадров")

        cap.release()

        for _ in range(self.num_workers):
            self.input_queue.put((None, None))

    def process_frames_worker(self, worker_id):
        try:
            model = load_model()
            self.ready_queue.put((worker_id, ""))
        except Exception as err:
            self.ready_queue.put((worker_id, str(err)))
            return

        while True:
            frame_id, frame = self.input_queue.get()

            if frame is None:
                break

            try:
                results = model(frame, verbose=False)
                processed_frame = results[0].plot()
                self.output_queue.put((frame_id, processed_frame))
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                self.output_queue.put((frame_id, frame))

        print(f"Worker {worker_id} finished")

    def write_video(self):
        """Запись видео с восстановлением порядка"""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

        if not out.isOpened():
            raise RuntimeError(f"Не удалось создать видео: {self.output_path}")

        processed_count = 0

        while processed_count < self.total_frames:
            try:
                frame_id, frame = self.output_queue.get(timeout=1)

                # Буферизируем кадры для восстановления порядка
                self.frame_buffer[frame_id] = frame

                # Выдаем кадры по порядку
                while self.next_frame_id in self.frame_buffer:
                    out.write(self.frame_buffer[self.next_frame_id])
                    del self.frame_buffer[self.next_frame_id]
                    self.next_frame_id += 1
                    processed_count += 1

                    if processed_count % 30 == 0:
                        print(f"Записано {processed_count}/{self.total_frames} кадров")

            except queue.Empty:
                print("Ожидание кадров...")
                continue

        out.release()
        print("Запись видео завершена")

    def run_single_thread(self):
        """Однопоточная обработка"""
        print("Запуск однопоточной обработки...")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.video_path}")

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

        if not out.isOpened():
            raise RuntimeError(f"Не удалось создать видео: {self.output_path}")

        model = load_model()

        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            processed_frame = results[0].plot()
            out.write(processed_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Обработано {frame_count}/{self.total_frames} кадров")

        elapsed = time.time() - start_time

        cap.release()
        out.release()

        print(
            f"Однопоточная обработка: {elapsed:.2f} сек, FPS: {frame_count/elapsed:.2f}"
        )
        return elapsed

    def run_multi_thread(self):
        """Многопоточная обработка"""
        print(f"Запуск многопоточной обработки с {self.num_workers} потоками...")

        # Запускаем потоки-воркеры
        workers = []
        for i in range(self.num_workers):
            worker = threading.Thread(target=self.process_frames_worker, args=(i,))
            worker.start()
            workers.append(worker)

        error_message = ""
        for _ in range(self.num_workers):
            _, error = self.ready_queue.get()
            if error and not error_message:
                error_message = error

        if error_message:
            for _ in workers:
                self.input_queue.put((None, None))

            for worker in workers:
                worker.join()

            raise RuntimeError(error_message)

        start_time = time.time()

        reader_thread = threading.Thread(target=self.read_video)
        reader_thread.start()

        self.write_video()

        reader_thread.join()
        for worker in workers:
            worker.join()

        elapsed = time.time() - start_time

        print(
            f"Многопоточная обработка ({self.num_workers} потоков): {elapsed:.2f} сек, FPS: {self.total_frames/elapsed:.2f}"
        )
        return elapsed


def run_benchmark(video_path, output_path):
    rows = []

    single_output = make_output_name(output_path, "single")
    processor = VideoProcessor(video_path, single_output, 1)
    processor.get_video_info()
    single_time = processor.run_single_thread()
    single_fps = processor.total_frames / single_time if single_time > 0 else 0.0

    rows.append({
        "mode": "single",
        "workers": 1,
        "time_s": f"{single_time:.6f}",
        "fps": f"{single_fps:.3f}",
        "speedup": "1.000",
        "output": single_output,
    })

    for workers in BENCHMARK_WORKERS:
        multi_output = make_output_name(output_path, str(workers))
        processor = VideoProcessor(video_path, multi_output, workers)
        processor.get_video_info()
        elapsed = processor.run_multi_thread()
        fps = processor.total_frames / elapsed if elapsed > 0 else 0.0
        speedup = single_time / elapsed if elapsed > 0 else 0.0

        rows.append({
            "mode": "multi",
            "workers": workers,
            "time_s": f"{elapsed:.6f}",
            "fps": f"{fps:.3f}",
            "speedup": f"{speedup:.3f}",
            "output": multi_output,
        })

    csv_path = make_csv_name(output_path)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "workers", "time_s", "fps", "speedup", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nЗамеры сохранены в {csv_path}")


def main():
    args = parse_args()

    if args.num_workers < 1:
        raise RuntimeError("Количество потоков должно быть положительным")

    if args.workmode == "benchmark":
        run_benchmark(args.path, args.name)
        return

    processor = VideoProcessor(args.path, args.name, args.num_workers)
    processor.get_video_info()

    if args.workmode == "single":
        processor.run_single_thread()
    else:
        processor.run_multi_thread()

    print(f"\nГотово! Результат сохранен в {args.name}")


if __name__ == "__main__":
    main()
