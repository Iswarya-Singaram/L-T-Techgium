import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import webbrowser  # New Import
from pyzbar import pyzbar

# --- Environment Fix ---
os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'

# --- Configuration ---
ENGINE_PATH = os.path.expanduser("~/qr.engine")
INPUT_SHAPE = (640, 640)
CONF_THRES = 0.5

# Keep track of opened links to avoid spamming browser tabs
opened_links = set()

def load_engine(path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # Allocate GPU Buffers
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    cap = cv2.VideoCapture(0)
    print("QR Detection & Auto-Navigator Started...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        orig_h, orig_w = frame.shape[:2]
        img = cv2.resize(frame, INPUT_SHAPE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        
        # --- Inference ---
        np.copyto(h_input, img.ravel())
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        # --- Post-processing ---
        detections = h_output.reshape(-1, 6)

        for det in detections:
            confidence = det[4] * det[5]
            if confidence > CONF_THRES:
                cx, cy, w, h = det[:4]
                
                x1 = int((cx - w/2) * orig_w / INPUT_SHAPE[0])
                y1 = int((cy - h/2) * orig_h / INPUT_SHAPE[1])
                x2 = int((cx + w/2) * orig_w / INPUT_SHAPE[0])
                y2 = int((cy + h/2) * orig_h / INPUT_SHAPE[1])

                crop = frame[max(0, y1-10):min(orig_h, y2+10), max(0, x1-10):min(orig_w, x2+10)]
                
                if crop.size > 0:
                    decoded_objs = pyzbar.decode(crop)
                    for obj in decoded_objs:
                        qr_data = obj.data.decode("utf-8")
                        
                        # --- AUTO-NAVIGATE LOGIC ---
                        if qr_data.startswith("http") and qr_data not in opened_links:
                            print(f"Opening Link: {qr_data}")
                            webbrowser.open(qr_data) # This opens the default browser
                            opened_links.add(qr_data)
                        elif qr_data not in opened_links:
                            print(f"Decoded Text: {qr_data}")
                            opened_links.add(qr_data)
                        
                        cv2.putText(frame, qr_data, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("QR Auto-Navigator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
