import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import easyocr
import firebase_admin
from firebase_admin import credentials, db

# ===== Initialize Firebase =====
cred = credentials.Certificate("firebase_key.json")  # Replace with your actual JSON file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'YOUR_DATABASE_KEY'
})

# ===== Initialize Models =====
plate_detector = YOLO('best.pt')  
reader = easyocr.Reader(['en'])       

# ===== CSV Setup =====
csv_path = 'parking_data.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['Plate_Number', 'Entry_Time', 'Exit_Time', 'Parking_Slot'])

# ===== Camera Function with Real-Time Detection =====
def capture_image():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
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
        
        results = plate_detector(frame)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        
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
    
    results = plate_detector(image)
    plates = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            plate_img = image[y1:y2, x1:x2]
            
            ocr_result = reader.readtext(plate_img)
            if ocr_result:
                plate_text = ocr_result[0][1].strip().replace(" ", "")
                plates.append(plate_text)
    
    return plates[0] if plates else None

# ===== Parking Slot Management with Firebase =====
def allocate_slot(plate_number):
    global df
    ref = db.reference("parking_slots")
    slots = ref.get()

    if slots is None or isinstance(slots, list):
        slots = {}

    max_slots = 10
    used_slots = [int(slot.split()[1]) for slot in slots.keys() if slots[slot]]

    available_slots = [s for s in range(1, max_slots+1) if s not in used_slots]
    
    if not available_slots:
        return "No slots available!"
    
    new_slot = f"Slot {available_slots[0]}"
    entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to Firebase
    ref.child(new_slot).child(plate_number).set({
        "Entry_Time": entry_time,
        "Exit_Time": None
    })

    # Save to CSV
    new_entry = {
        'Plate_Number': plate_number,
        'Entry_Time': entry_time,
        'Exit_Time': None,
        'Parking_Slot': available_slots[0]
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    
    return f"✅ Allocated Slot: {new_slot} for {plate_number}"

def clear_slot(plate_number):
    global df
    ref = db.reference("parking_slots")
    slots = ref.get()

    if slots is None or isinstance(slots, list):
        slots = {}

    for slot, cars in slots.items():
        if plate_number in cars:
            car_data = cars[plate_number]

            if "Exit_Time" not in car_data or car_data["Exit_Time"] is None:
                exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Update Firebase
                ref.child(slot).child(plate_number).update({
                    "Exit_Time": exit_time
                })

                # Update CSV
                mask = (df['Plate_Number'] == plate_number) & (df['Exit_Time'].isna())
                df.loc[mask, 'Exit_Time'] = exit_time
                df.to_csv(csv_path, index=False)

                return f"✅ Car {plate_number} exited from {slot}."
            else:
                return f"⚠️ Car {plate_number} already exited from {slot}."

    return "❌ Car not found in any slot!"

# ===== Main Program =====
if __name__ == "__main__":
    print("\n=== 🚗 Smart Parking Management System 🚗 ===")
    while True:
        action = input("\nEnter 'IN' for entry, 'OUT' for exit, or 'EXIT' to quit: ").strip().upper()
        
        if action == 'IN':
            input("Press Enter to open camera...")
            image = capture_image()
            if image is not None:
                plate = detect_plate(image)
                if plate:
                    print(f"🔍 Detected Plate: {plate}")
                    print(allocate_slot(plate))
                else:
                    print("❌ No plate detected. Please try again!")
        
        elif action == 'OUT':
            input("Press Enter to open camera...")
            image = capture_image()
            if image is not None:
                plate = detect_plate(image)
                if plate:
                    print(f"🔍 Detected Plate: {plate}")
                    print(clear_slot(plate))
                else:
                    print("❌ No plate detected. Please try again!")
        
        elif action == 'EXIT':
            print("🚪 System shutdown.")
            break
        
        else:
            print("⚠️ Invalid command! Please enter IN/OUT/EXIT.") 
