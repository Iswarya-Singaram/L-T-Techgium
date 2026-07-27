import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# --- Environment Fix ---
os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'

# --- Configuration ---
ENGINE_PATH = os.path.expanduser("~/hotspot.engine")
INPUT_SHAPE = (640, 640)
CONF_THRES = 0.45 

CLASSES = ['hotspot', 'square', 'target', 'triangle']
COLORS = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

def load_engine(path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # Allocate Buffers
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    cap = cv2.VideoCapture(0)
    
    # --- WINDOW SETUP FOR FULL SCREEN ---
    window_name = "Hotspot Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Hotspot Detection Active. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Pre-processing
        orig_h, orig_w = frame.shape[:2]
        img = cv2.resize(frame, INPUT_SHAPE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        
        # 2. Inference
        np.copyto(h_input, img.ravel())
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        # 3. Post-processing
        num_outputs = 4 + len(CLASSES)
        output = h_output.reshape(num_outputs, -1).T

        for row in output:
            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > CONF_THRES:
                cx, cy, w, h = row[:4]
                x1 = int((cx - w/2) * orig_w / INPUT_SHAPE[0])
                y1 = int((cy - h/2) * orig_h / INPUT_SHAPE[1])
                x2 = int((cx + w/2) * orig_w / INPUT_SHAPE[0])
                y2 = int((cy + h/2) * orig_h / INPUT_SHAPE[1])

                color = COLORS[class_id]
                label = f"{CLASSES[class_id]} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Use the specific window name to display
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
