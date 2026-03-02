import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# --- Configuration ---
ENGINE_PATH = os.path.expanduser("~/ppe.engine")
INPUT_SHAPE = (640, 640)
CONF_THRES = 0.5

# Your Specific PPE Classes
CLASSES = [
    'Boots', 
    'Gloves', 
    'Mask', 
    'Safety-Helmet', 
    'Safety-Vest', 
    'Safety-Wearpack'
]

# Distinct colors for each class (B, G, R)
COLORS = [
    (255, 100, 0),   # Boots (Blue-ish)
    (0, 255, 255),   # Gloves (Yellow)
    (255, 0, 255),   # Mask (Magenta)
    (0, 255, 0),     # Safety-Helmet (Green)
    (0, 165, 255),   # Safety-Vest (Orange)
    (200, 200, 200)  # Safety-Wearpack (Grey)
]

def load_engine(path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():
    if not os.path.exists(ENGINE_PATH):
        print(f"ERROR: Engine file not found at {ENGINE_PATH}")
        return

    # 1. Initialize TensorRT
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # 2. Allocate GPU Buffers
    # The engine has bindings (input/output). We map them to GPU memory.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    # 3. Setup Camera (Try 0, then 1 if 0 fails)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    print(f"PPE Detection Started. Classes: {len(CLASSES)}")
    print("Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- PRE-PROCESSING ---
        original_h, original_w = frame.shape[:2]
        # Resize to model input size
        input_img = cv2.resize(frame, INPUT_SHAPE)
        # Convert BGR to RGB, then to CHW (Channel, Height, Width) format
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.transpose((2, 0, 1)).astype(np.float32)
        input_img /= 255.0  # Normalize to [0, 1]
        
        # --- INFERENCE ---
        np.copyto(h_input, input_img.ravel())
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        # --- POST-PROCESSING ---
        # Assuming YOLOv5 standard output [1, 25200, 85] or similar
        # We flatten it here; you may need to adjust the reshape based on your model's specific output
        detections = h_output.reshape(-1, 5 + len(CLASSES)) 

        for det in detections:
            obj_conf = det[4]
            if obj_conf > CONF_THRES:
                # Find the class with the highest probability
                class_scores = det[5:]
                class_id = np.argmax(class_scores)
                final_score = obj_conf * class_scores[class_id]
                
                if final_score > CONF_THRES:
                    # Rescale coordinates to original frame size
                    x, y, w, h = det[:4]
                    x1 = int((x - w/2) * original_w / INPUT_SHAPE[0])
                    y1 = int((y - h/2) * original_h / INPUT_SHAPE[1])
                    x2 = int((x + w/2) * original_w / INPUT_SHAPE[0])
                    y2 = int((y + h/2) * original_h / INPUT_SHAPE[1])

                    # --- VISUALIZATION ---
                    label = CLASSES[class_id] if class_id < len(CLASSES) else "Unknown"
                    color = COLORS[class_id] if class_id < len(COLORS) else (0, 255, 0)
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Draw label background
                    label_str = f"{label} {final_score:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
                    cv2.putText(frame, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show Output
        cv2.imshow("Jetson Nano PPE Surveillance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
