import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import easyocr

# ===== Initialize Models =====
plate_detector = YOLO('best.pt')  # Auto-downloads YOLOv8n model
reader = easyocr.Reader(['en'])       # Initialize EasyOCR

# ===== CSV Setup =====
csv_path = 'parking_data.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['Plate_Number', 'Entry_Time', 'Exit_Time', 'Parking_Slot'])

# ===== Camera Function with Real-Time Detection =====
def capture_image():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-specific fix
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return None
    
    print("Camera opened. Press 's' to capture or 'q' to quit...")
    captured_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect license plates and draw bounding boxes
        results = plate_detector(frame)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        
        cv2.imshow("Camera - Press 's' to CAPTURE", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            print("Image captured!")
            captured_frame = frame.copy()
            break
        elif key == ord('q'):
            captured_frame = None
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_frame

# ===== License Plate Text Extraction =====
def detect_plate(image):
    if image is None:
        return None
    
    # Detect plates using YOLOv8
    results = plate_detector(image)
    plates = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            plate_img = image[y1:y2, x1:x2]
            
            # Extract text with EasyOCR
            ocr_result = reader.readtext(plate_img)
            if ocr_result:
                plate_text = ocr_result[0][1].strip().replace(" ", "")
                plates.append(plate_text)
    
    return plates[0] if plates else None

# ===== Parking Slot Management =====
def allocate_slot(plate_number):
    global df
    # Check if car is already parked
    active_parked = df[df['Exit_Time'].isna()]
    if plate_number in active_parked['Plate_Number'].values:
        return "Car already parked!"
    
    # Assign first available slot
    max_slots = 10
    used_slots = active_parked['Parking_Slot'].tolist()
    available_slots = [s for s in range(1, max_slots+1) if s not in used_slots]
    
    if not available_slots:
        return "No slots available!"
    
    new_slot = available_slots[0]
    new_entry = {
        'Plate_Number': plate_number,
        'Entry_Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Exit_Time': None,
        'Parking_Slot': new_slot
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return f"Allocated Slot: {new_slot}"

def clear_slot(plate_number):
    global df
    # Find active parking entry
    mask = (df['Plate_Number'] == plate_number) & (df['Exit_Time'].isna())
    if df[mask].empty:
        return "Car not found!"
    
    df.loc[mask, 'Exit_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(csv_path, index=False)
    return f"Slot {df[mask]['Parking_Slot'].values[0]} cleared."

# ===== Main Program =====
if __name__ == "__main__":
    print("\n=== Smart Parking Management System ===")
    while True:
        action = input("\nEnter 'IN' for entry, 'OUT' for exit, or 'EXIT' to quit: ").strip().upper()
        
        if action == 'IN':
            input("Press Enter to open camera...")
            image = capture_image()
            if image is not None:
                plate = detect_plate(image)
                if plate:
                    print(f"Detected Plate: {plate}")
                    print(allocate_slot(plate))
                else:
                    print("No plate detected. Please try again!")
        
        elif action == 'OUT':
            input("Press Enter to open camera...")
            image = capture_image()
            if image is not None:
                plate = detect_plate(image)
                if plate:
                    print(f"Detected Plate: {plate}")
                    print(clear_slot(plate))
                else:
                    print("No plate detected. Please try again!")
        
        elif action == 'EXIT':
            print("System shutdown.")
            break
        
        else:
            print("Invalid command! Please enter IN/OUT/EXIT.")