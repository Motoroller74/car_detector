import cv2
import pytesseract
import csv
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
import os
import shutil
from time import sleep

# ===== НАСТРОЙКИ =====
PLATES_CSV = "plates.csv"
LOGS_DIR = "logs"
PHOTOS_DIR = "photos"    # Папка для хранения фото по датам
PHOTO_MAX_DAYS = 14      # Удалять фото старше 14 дней

# ===== ФУНКЦИИ =====
def load_allowed_plates():
    """Загружает номера из CSV-файла."""
    plates = []
    try:
        with open(PLATES_CSV, "r", encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                plates.append({
                    "номер": row["Номер"].strip(),
                    "модель": row["Модель"].strip(),
                    "владелец": row["Владелец"].strip()
                })
        print("Загруженные номера:", plates)  # Для отладки
        return plates
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {PLATES_CSV} не найден!")
        return []
    except KeyError as e:
        print(f"❌ Ошибка в структуре CSV: отсутствует колонка {e}")
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

def save_photo(image):
    """Сохраняет фото в папку по дате."""
    today = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(PHOTOS_DIR, today)
    os.makedirs(date_dir, exist_ok=True)
    
    filename = f"{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.jpg"
    filepath = os.path.join(date_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath

def cleanup_old_photos(max_days=PHOTO_MAX_DAYS):
    """Удаляет папки с фото старше max_days дней."""
    now = datetime.now()
    cutoff_date = now - timedelta(days=max_days)
    
    if not os.path.exists(PHOTOS_DIR):
        return
    
    for dir_name in os.listdir(PHOTOS_DIR):
        dir_path = os.path.join(PHOTOS_DIR, dir_name)
        if os.path.isdir(dir_path):
            try:
                dir_date = datetime.strptime(dir_name, "%Y-%m-%d")
                if dir_date < cutoff_date:
                    shutil.rmtree(dir_path)
                    print(f"🗑️ Удалена старая папка: {dir_path}")
            except ValueError:
                continue

def capture_and_process_plate():
    """Делает фото, сохраняет его и распознает номер."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log_event("Ошибка камеры")
        return None
    
    print("Нажмите 's' чтобы сделать фото, 'q' чтобы выйти...")
    while True:
        ret, frame = cap.read()
        if not ret:
            log_event("Ошибка съемки")
            break
        
        cv2.imshow("Камера (Нажмите 's' для снимка)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Клавиша 's' для снимка
            photo_path = save_photo(frame)
            print(f"📸 Фото сохранено: {photo_path}")
            cap.release()
            cv2.destroyAllWindows()
            plate_number = recognize_plate(frame, photo_path)
            if plate_number:
                return plate_number
            else:
                print("❌ Не удалось распознать номер!")
        
        elif key == ord('q'):  # Клавиша 'q' для выхода
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def open_img(img_path):
    carplate_img = cv2.imread(img_path)
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
    #plt.axis('off')
    #plt.imshow(carplate_img)
    # plt.show()

    return carplate_img


def carplate_extract(image, carplate_haar_cascade):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=6)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y+15:y+h-10, x+15:x+w-20]

    return carplate_img


def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    #plt.axis('off')
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized_image


def recognize_plate(image, image_path):
    """Распознает номер с изображения."""
    print(image_path)
    try:
        carplate_img_rgb = open_img(image_path)
        carplate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

        carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade)
        carplate_extract_img = enlarge_img(carplate_extract_img, 250)
        # plt.imshow(carplate_extract_img)
        # plt.show()

        carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
        # plt.axis('off')
        # plt.imshow(carplate_extract_img_gray, cmap='gray')
        # plt.show()

        plate_text = pytesseract.image_to_string(
            carplate_extract_img_gray,
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789')

        if plate_text:
        # Преобразуем строку в список для изменения символов
            plate_list = list(plate_text)
            
            for i, w in enumerate(plate_list):
                if i in [0, 4, 5] and w == '0':  # Буквенные позиции (A, E, K, M, H, O, P, C, T, Y, X)
                    plate_list[i] = 'O'
                elif i in [1, 2, 3, 6, 7] and w == 'O':  # Цифровые позиции
                    plate_list[i] = '0'
        
            # Собираем обратно в строку и удаляем возможные пробелы/переносы
            plate_text = ''.join(plate_list).strip()
        return plate_text.replace(" ","") if plate_text else None
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
    print("🚀 Система запущена. Нажмите Enter для сканирования номера...")
    while True:
        input("Нажмите Enter для старта или Ctrl+C для выхода...")
        plate_number = capture_and_process_plate()
        
        if plate_number:
            print(f"🔍 Найден номер: {plate_number}")
            plate_info = next(
                (p for p in ALLOWED_PLATES if p["номер"] == plate_number),
                None
            )
            
            if plate_info:
                log_event("Доступ разрешен", plate_info)
                print(f"🟢 Доступ разрешен: {plate_info['модель']} ({plate_info['владелец']})")
                print("[Шлагбаум открыт]")  # Вместо GPIO
                sleep(2)
                print("[Шлагбаум закрыт]")
            else:
                log_event("Доступ запрещен", {"номер": plate_number})
                print("🔴 Доступ запрещен: номер не найден в базе!")
        
except KeyboardInterrupt:
    print("🛑 Остановка системы...")
finally:
    cv2.destroyAllWindows()
    
    