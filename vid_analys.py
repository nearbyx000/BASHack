import cv2
import numpy as np
import time
import sys
import os
import random
from enum import Enum
import asyncio
import threading
import grpc
import json
import datetime

try:
    from . import video_pb2
    from . import video_pb2_grpc
except ImportError:
    print("Предупреждение: Не удалось импортировать video_pb2.py или video_pb2_grpc.py.")
    class MockVideoPb2:
        class Frame:
            def __init__(self, data=None, encoding=None):
                pass
    class MockVideoPb2Grpc:
        class VideoStreamServiceStub:
            def __init__(self, channel):
                pass
    video_pb2 = MockVideoPb2()
    video_pb2_grpc = MockVideoPb2Grpc()

try:
    from ultralytics import YOLO
except ImportError:
    print("Ошибка: Библиотека 'ultralytics' не найдена.")
    print("Пожалуйста, установите ее, выполнив: pip install ultralytics")
    sys.exit(1)

import msgpackrpc
from PIL import Image, ImageDraw, ImageFont

class CaptureType(Enum):
    color = 0
    thermal = 1
    depth = 2
    spectrum_color = 3
    spectrum_NIR = 4
    spectrum_SWIR = 5
    spectrum_RE = 6
    spectrum_R = 7
    spectrum_G = 8
    spectrum_B = 9  # Исправлено: B вместо spectrum_B

def post_process(image, gamma=1.0, new_size=(800, 600), saturation=1.0, contrast=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    image = cv2.LUT(image, table)
    if new_size is not None:
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    if saturation != 1.0:
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return image

class VideoStreamSender:
    def __init__(self, camera_id=0, rate=30):
        self.client = SimClient()
        self.camera_id = camera_id
        self.streaming = False
        self.rate = rate

    async def generate_frames(self):
        while self.streaming:
            frame = self.client.get_camera_capture(camera_id=self.camera_id, type=CaptureType.color)
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield video_pb2.Frame(data=buffer.tobytes(), encoding="jpeg")
            await asyncio.sleep(1 / self.rate)

    async def stream(self, port):
        async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
            stub = video_pb2_grpc.VideoStreamServiceStub(channel)
            await stub.StreamFrames(self.generate_frames())

sender_instance = None
thread_instance = None

class SimClient:
    def __init__(self, address="127.0.0.1", port=8080):
        self.address = address
        self.port = port
        self.rpc_client = msgpackrpc.Client(msgpackrpc.Address(self.address, self.port), 
                                            timeout=10, 
                                            pack_encoding='utf-8', 
                                            unpack_encoding='utf-8')
        self.streaming = False
    
    def __del__(self):
        self.close_connection()

    def close_connection(self):
        if self.is_connected():
            self.rpc_client.close()
        
    def add_noise(self, image):
        noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image

    def add_artifacts(self, image):
        h, w, _ = image.shape
        for _ in range(random.randint(1, 7)):
            y_line = np.random.randint(0, h)
            width = np.random.randint(3, 10)
            line_end = min(y_line + width, h)
            image[y_line:line_end, :] = np.random.randint(0, 255, size=(line_end - y_line, w, 3), dtype=np.uint8)
        return image

    def is_connected(self):
        result = True
        try:
            result = self.rpc_client.call('ping')
        except:
            result = False
        return result

    def get_laser_scan(self, angle_min=-np.pi/2, angle_max=np.pi/2, range_min=0.1, range_max=30, num_ranges=30, is_clear=False, range_error=0.15):
        laser_scan_data = self.rpc_client.call('getLaserScan', angle_min, angle_max, range_min, range_max, num_ranges)
        if not is_clear and len(laser_scan_data) == num_ranges:
            noise = np.random.normal(0, range_error, num_ranges)
            laser_scan_data += noise
        return laser_scan_data
    
    def get_radar_point(self, radar_id=0, base_angle=45, range_min=0.15, range_max=5, is_clear=True, range_error=0.15, angle_error=0.015):
        radar_point = self.rpc_client.call('getRadarData', radar_id, base_angle, range_min, range_max)
        radar_point[1] = -radar_point[1]
        if not is_clear:
            range_noise = np.random.normal(0, range_error, 1)
            radar_point[0] += range_noise
            angle_noise = np.random.normal(0, angle_error, 2)
            radar_point[1:] += angle_noise 
        return radar_point
    
    def get_range_data(self, rangefinder_id=0, range_min=0.15, range_max=10, is_clear=True, range_error=0.15):
        range_point = self.rpc_client.call('getRangefinderData', rangefinder_id, range_min, range_max)
        if not is_clear:
            noise = np.random.normal(0, range_error, 1)
            range_point += noise
        return range_point

    def set_led_intensity(self, led_id=0, new_intensity=0.5):
        self.rpc_client.call('setLedIntensity', led_id, new_intensity)

    def set_led_state(self, led_id=0, new_state=True):
        self.rpc_client.call('setLedState', led_id, new_state)

    def get_kinematics_data(self):
        return self.rpc_client.call("getKinematicsData")
    
    def call_event_action(self):
        try:
            return self.rpc_client.call("callEventAction")
        except:
            return False

    def start_streaming(self, port, camera_id=0, rate=30):
        global sender_instance, thread_instance
        if sender_instance is not None and sender_instance.streaming:
            print("[INFO] Streaming already running")
            return
        sender_instance = VideoStreamSender(camera_id, rate)
        sender_instance.streaming = True
        def run_async():
            asyncio.run(sender_instance.stream(port))
        thread_instance = threading.Thread(target=run_async, daemon=True)
        thread_instance.start()
        print(f"[INFO] Started streaming to port {port}")

    def stop_streaming(self):
        global sender_instance
        if sender_instance:
            sender_instance.streaming = False
            print("[INFO] Stopped streaming")
        else:
            print("[WARN] No active streaming session")

    def get_camera_capture(self, camera_id=0, type=CaptureType.color):
        pp_index = 0
        parameter = 0
        if type == CaptureType.color:
            pp_index = 0
            parameter = 0
        elif type == CaptureType.thermal:
            pp_index = 1
            parameter = 0
        elif type == CaptureType.depth:
            pp_index = 2
            parameter = 0
        elif type == CaptureType.spectrum_color:
            pp_index = 3
            parameter = 0
        elif type == CaptureType.spectrum_NIR:
            pp_index = 3
            parameter = 1
        elif type == CaptureType.spectrum_SWIR:
            pp_index = 3
            parameter = 2
        elif type == CaptureType.spectrum_RE:
            pp_index = 3
            parameter = 3
        elif type == CaptureType.spectrum_R:
            pp_index = 3
            parameter = 4
        elif type == CaptureType.spectrum_G:
            pp_index = 3
            parameter = 5
        elif type == CaptureType.spectrum_B:  # Исправлено
            pp_index = 3
            parameter = 6
        raw_image = self.rpc_client.call('getCameraCapture', camera_id, pp_index, parameter)
        if len(raw_image) > 1:
            print(f"Image size: {len(raw_image)}")
            cv2_image = np.frombuffer(bytes(raw_image), dtype=np.uint8).reshape((360, 480, 4))
            result = post_process(cv2_image, gamma=1.0, new_size=(640, 480), saturation=1.05, contrast=1)
            return result

MODEL_PATH = 'best.pt'
CAMERA_ID = 1
FRAME_DELAY = 0.05
SIM_ADDRESS = "127.0.0.1"
SIM_PORT = 8080
FONT_PATH = "C:/Windows/Fonts/arial.ttf"
FONT_SIZE = 30
DETECT_EVERY_N_FRAMES = 5  # Новая константа: детекция каждые 5 кадров

total_sunflowers_detected = 0

def main():
    global total_sunflowers_detected
    print(f"Загрузка модели YOLO из: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        sys.exit(1)

    print(f"Подключение к симулятору по адресу: {SIM_ADDRESS}:{SIM_PORT}")
    client = SimClient(address=SIM_ADDRESS, port=SIM_PORT)
    if not client.is_connected():
        print("Ошибка: Не удалось подключиться к симулятору.")
        sys.exit(1)
    print("Успешно подключено к симулятору.")

    cv2.namedWindow("Обнаружение подсолнухов", cv2.WINDOW_NORMAL)
    font_pil = None
    try:
        font_pil = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        print(f"Шрифт {FONT_PATH} успешно загружен.")
    except IOError:
        print(f"Предупреждение: Не удалось загрузить шрифт {FONT_PATH}.")
    except Exception as e:
        print(f"Неизвестная ошибка при загрузке шрифта: {e}.")

    frame_count = 0
    start_time = time.time()
    detect_counter = 0
    frame_number = 0
    output_data = []  # Для сохранения результатов в JSON
    previous_annotated_frame = None

    try:
        while True:
            frame = client.get_camera_capture(camera_id=CAMERA_ID, type=CaptureType.color)
            if frame is None:
                print("Не удалось получить кадр с камеры.")
                time.sleep(1)
                continue

            frame_number += 1
            detect_counter += 1
            annotated_frame = frame.copy()

            # Выполняем детекцию только каждые DETECT_EVERY_N_FRAMES
            current_detections = []
            if detect_counter >= DETECT_EVERY_N_FRAMES:
                detect_counter = 0
                results = model(frame, verbose=False)
                for r in results:
                    num_detections_in_frame = len(r.boxes)
                    total_sunflowers_detected += num_detections_in_frame
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        cls = int(box.cls[0])
                        class_name = model.names[cls] if hasattr(model, 'names') and cls in model.names else f"Class {cls}"
                        color = (0, 255, 0)
                        thickness = 2
                        font_scale_cv2 = 0.7
                        font_thickness_cv2 = 2
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        label_text = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale_cv2, color, font_thickness_cv2)
                        # Собираем данные для JSON
                        current_detections.append({
                            "class": class_name,
                            "confidence": float(conf),
                            "bbox": [x1, y1, x2, y2]
                        })
                previous_annotated_frame = annotated_frame
            else:
                # Используем предыдущий аннотированный кадр
                if previous_annotated_frame is not None:
                    annotated_frame = previous_annotated_frame

            # Сохраняем данные кадра в JSON
            output_data.append({
                "frame_number": frame_number,
                "timestamp": datetime.datetime.now().isoformat(),
                "total_sunflowers": total_sunflowers_detected,
                "detections": current_detections
            })

            # Отображаем счётчик
            count_text = f"Всего подсолнухов: {total_sunflowers_detected}"
            text_position = (10, 30)
            font_color_bgr = (0, 0, 255)
            if font_pil:
                pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                font_color_rgb = (font_color_bgr[2], font_color_bgr[1], font_color_bgr[0])
                draw.text(text_position, count_text, font=font_pil, fill=font_color_rgb)
                annotated_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(annotated_frame, count_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, font_color_bgr, 2)

            cv2.imshow("Обнаружение подсолнухов", annotated_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                print(f"FPS: {fps:.2f}")
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(FRAME_DELAY)

    except KeyboardInterrupt:
        print("\nОбнаружено прерывание пользователем (Ctrl+C).")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        # Сохраняем результаты в JSON
        with open("sunflowers_detection.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print("Результаты сохранены в sunflowers_detection.json")
        client.close_connection()
        cv2.destroyAllWindows()
        print("Скрипт завершен.")

if __name__ == "__main__":
    main()