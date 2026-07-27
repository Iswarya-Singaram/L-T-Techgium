import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTDetector:
    def __init__(self, engine_path):
        # 1. Initialize Engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        # 2. Identify Input/Output Shapes
        self.input_shape = self.engine.get_binding_shape(0)
        self.img_h, self.img_w = self.input_shape[2], self.input_shape[3]
        print(f"--- Model Loaded ---")
        print(f"Input required: {self.img_w}x{self.img_h}")

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def detect(self, frame):
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess: Resize, BGR to RGB, Normalize
        blob = cv2.resize(frame, (self.img_w, self.img_h))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)

        # Run Inference
        np.copyto(self.inputs[0]['host'], blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Reshape output based on engine binding
        raw_output = self.outputs[0]['host'].reshape(self.engine.get_binding_shape(1))
        return self.parse_yolo(raw_output, orig_w, orig_h)

    def parse_yolo(self, output, orig_w, orig_h):
        # Remove batch dim (1, 5, 8400) -> (5, 8400)
        predictions = np.squeeze(output)
        
        # If shape is [5, 8400], transpose to [8400, 5]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes, confs = [], []

        for pred in predictions:
            conf = pred[4] # Assuming 5th element is confidence (YOLO standard)
            if conf > 0.45: # Threshold
                x, y, w, h = pred[0:4]
                
                # Scale coordinates to original image size
                # YOLO output is center_x, center_y, width, height
                l = int((x - w/2) * (orig_w / self.img_w))
                t = int((y - h/2) * (orig_h / self.img_h))
                r = int(l + (w * (orig_w / self.img_w)))
                b = int(t + (h * (orig_h / self.img_h)))
                
                boxes.append([l, t, r, b])
                confs.append(float(conf))

        if not boxes:
            return []

        # Apply Manual Non-Maximum Suppression
        return self.manual_nms(np.array(boxes), np.array(confs), 0.45)

    def manual_nms(self, boxes, scores, iou_threshold):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return boxes[keep]

# --- Main Runtime ---
if __name__ == "__main__":
    detector = TensorRTDetector("human.engine")
    cap = cv2.VideoCapture(0)

    print("Starting video... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        human_boxes = detector.detect(frame)

        # Draw the results
        for box in human_boxes:
            l, t, r, b = box.astype(int)
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, "Human", (l, t - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Human Detection Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
