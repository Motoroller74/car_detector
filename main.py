import cv2
import pytesseract
import RPi.GPIO as GPIO
import csv
from datetime import datetime, timedelta
import os
import shutil
from time import sleep

# ===== НАСТРОЙКИ =====
RELAY_PIN = 17           # GPIO для реле/шлагбаума
PIR_PIN = 4              # GPIO для датчика движения (PIR)
PLATES_CSV = "plates.csv"
LOGS_DIR = "logs"
PHOTOS_DIR = "photos"    # Папка для хранения фото по датам
PHOTO_MAX_DAYS = 14      # Удалять фото старше 14 дней

# ===== ИНИЦИАЛИЗАЦИЯ GPIO =====
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.setup(PIR_PIN, GPIO.IN)

# ===== ФУНКЦИИ =====
def load_allowed_plates():
    """Загружает номера из CSV-файла."""
    plates = []
    try:
        with open(PLATES_CSV, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                plates.append({
                    "номер": row["Номер"].strip(),
                    "модель": row.get("Модель", "").strip(),
                    "владелец": row.get("Владелец", "").strip()
                })
        return plates
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {PLATES_CSV} не найден!")
        return []

def log_event(status, plate_info=None):
    """Логирует события в файлы по датам."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = f"{LOGS_DIR}/{status.lower()}_{today}.log"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        "время": timestamp,
        "номер": plate_info["номер"] if plate_info else "Не распознан",
        "модель": plate_info.get("модель", "Нет данных") if plate_info else "Нет данных",
        "владелец": plate_info.get("владелец", "Нет данных") if plate_info else "Нет данных",
        "статус": status
    }
    
    with open(log_file, "a") as file:
        file.write(f"{log_data}\n")

def save_photo(image, plate_number):
    """Сохраняет фото в папку по дате."""
    today = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(PHOTOS_DIR, today)
    os.makedirs(date_dir, exist_ok=True)
    
    filename = f"{plate_number}_{datetime.now().strftime('%H%M%S')}.jpg"
    filepath = os.path.join(date_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath

def cleanup_old_photos(max_days=PHOTO_MAX_DAYS):
    """Удаляет папки с фото старше max_days дней."""
    now = datetime.now()
    cutoff_date = now - timedelta(days=max_days)
    
    for dir_name in os.listdir(PHOTOS_DIR):
        dir_path = os.path.join(PHOTOS_DIR, dir_name)
        if os.path.isdir(dir_path):
            dir_date = datetime.strptime(dir_name, "%Y-%m-%d")
            if dir_date < cutoff_date:
                shutil.rmtree(dir_path)
                print(f"🗑️ Удалена старая папка: {dir_path}")

def capture_and_process_plate():
    """Делает фото, сохраняет его и распознает номер."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log_event("Ошибка камеры")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        log_event("Ошибка съемки")
        return None
    
    plate_number = recognize_plate(frame)
    if plate_number:
        photo_path = save_photo(frame, plate_number)
        print(f"📸 Фото сохранено: {photo_path}")
        return plate_number
    else:
        log_event("Ошибка распознавания")
        return None

def recognize_plate(image):
    """Распознает номер с изображения."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plate_text = pytesseract.image_to_string(gray, config='--psm 8').strip()
        return plate_text if plate_text else None
    except Exception as e:
        print(f"❌ Ошибка распознавания: {e}")
        return None

# ===== ЗАГРУЗКА ДАННЫХ =====
ALLOWED_PLATES = load_allowed_plates()
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)
cleanup_old_photos()  # Очистка старых фото при запуске

# ===== ГЛАВНЫЙ ЦИКЛ =====
try:
    print("🚀 Система запущена. Ожидание движения...")
    while True:
        if GPIO.input(PIR_PIN):  # Если есть движение
            print("🚗 Обнаружена машина! Сканирую номер...")
            plate_number = capture_and_process_plate()
            
            if plate_number:
                print(f"🔍 Найден номер: {plate_number}")
                plate_info = next(
                    (p for p in ALLOWED_PLATES if p["номер"] == plate_number),
                    None
                )
                
                if plate_info:
                    log_event("Доступ разрешен", plate_info)
                    print(f"🟢 Открываю шлагбаум для {plate_info['модель']} ({plate_info['владелец']})")
                    GPIO.output(RELAY_PIN, GPIO.HIGH)
                    sleep(5)
                    GPIO.output(RELAY_PIN, GPIO.LOW)
                else:
                    log_event("Доступ запрещен", {"номер": plate_number})
                    print("🔴 Номер не найден в базе!")
            
            sleep(1)  # Задержка между проверками
        
        sleep(0.1)

except KeyboardInterrupt:
    print("🛑 Остановка системы...")
finally:
    GPIO.cleanup()