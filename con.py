import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import threading
import time
import webbrowser
from flask import Flask, Response, request, jsonify
from werkzeug.utils import secure_filename
from pyzbar import pyzbar

# --- Environment Fix ---
os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'

app = Flask(__name__)
UPLOAD_FOLDER = os.path.expanduser("~/uploaded_engines")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class CameraThread:
    """Independent thread to read camera frames as fast as possible."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Set buffer size to 1 for lowest latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
            time.sleep(0.01)

    def get_frame(self):
        return self.ret, self.frame

# Initialize Camera Thread
cam = CameraThread()

CONFIGS = {
    'ppe': {'classes': ['Boots', 'Gloves', 'Mask', 'Safety-Helmet', 'Safety-Vest', 'Safety-Wearpack'], 'colors': [(255, 100, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 165, 255), (200, 200, 200)]},
    'hotspot': {'classes': ['hotspot', 'square', 'target', 'triangle'], 'colors': [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0)]},
    'qr': {'classes': ['QR-Code'], 'colors': [(255, 0, 0)]},
    'human': {'classes': ['Human'], 'colors': [(0, 255, 0)]},
    'default': { 'classes': ['Object'], 'colors': [(0, 255, 0)] }
}

class Detector:
    def __init__(self):
        self.engine = None
        self.context = None
        self.cfx = cuda.Device(0).make_context()
        self.active_config = CONFIGS['default']
        self.active_type = 'default'
        self.opened_links = set()
        self.h_input, self.h_output = None, None
        self.d_input, self.d_output = None, None
        self.stream = None

    def load_engine(self, engine_path):
        self.cfx.push()
        try:
            filename = os.path.basename(engine_path).lower()
            self.active_type = next((t for t in ['ppe', 'hotspot', 'qr', 'human'] if t in filename), 'default')
            self.active_config = CONFIGS[self.active_type]
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                new_engine = runtime.deserialize_cuda_engine(f.read())
            
            self.h_input = cuda.pagelocked_empty(trt.volume(new_engine.get_binding_shape(0)), dtype=np.float32)
            self.h_output = cuda.pagelocked_empty(trt.volume(new_engine.get_binding_shape(1)), dtype=np.float32)
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            self.stream = cuda.Stream()
            self.context = new_engine.create_execution_context()
            self.engine = new_engine
            print(f"Loaded Engine: {self.active_type.upper()}")
        except Exception as e: print(f"Load Error: {e}")
        finally: self.cfx.pop()

    def apply_nms(self, boxes, scores, threshold=0.45):
        if len(boxes) == 0: return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= threshold)[0] + 1]
        return keep

    def detect(self, frame):
        if self.engine is None: return frame
        self.cfx.push()
        try:
            orig_h, orig_w = frame.shape[:2]
            img = cv2.resize(frame, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

            np.copyto(self.h_input, img.ravel())
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()

            predictions = np.squeeze(self.h_output.reshape(self.engine.get_binding_shape(1)))
            if predictions.shape[0] < predictions.shape[1]: predictions = predictions.T

            boxes, confs, class_ids = [], [], []
            for det in predictions:
                conf = det[4] if self.active_type != 'qr' else (det[4] * det[5])
                if conf > 0.5:
                    cid = np.argmax(det[5:]) if len(det) > 5 else 0
                    cx, cy, w, h = det[:4]
                    x1, y1 = int((cx - w/2) * orig_w / 640), int((cy - h/2) * orig_h / 640)
                    x2, y2 = int((cx + w/2) * orig_w / 640), int((cy + h/2) * orig_h / 640)
                    boxes.append([x1, y1, x2, y2]); confs.append(float(conf)); class_ids.append(cid)

            indices = self.apply_nms(np.array(boxes), np.array(confs)) if boxes else []
            for i in indices:
                bx, conf, cid = boxes[i], confs[i], class_ids[i]
                color = self.active_config['colors'][cid] if cid < len(self.active_config['colors']) else (0, 255, 0)
                
                if self.active_type == 'qr':
                    crop = frame[max(0, bx[1]):min(orig_h, bx[3]), max(0, bx[0]):min(orig_w, bx[2])]
                    if crop.size > 0:
                        for obj in pyzbar.decode(crop):
                            data = obj.data.decode("utf-8")
                            if data not in self.opened_links:
                                if data.startswith("http"): webbrowser.open(data)
                                self.opened_links.add(data)
                            cv2.putText(frame, data, (bx[0], bx[3]+20), 0, 0.5, (0, 255, 0), 2)

                cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
                cv2.putText(frame, f"{self.active_config['classes'][cid]}", (bx[0], bx[1]-10), 0, 0.5, color, 2)
        except Exception as e: print(f"Det Error: {e}")
        finally: self.cfx.pop()
        return frame

detector = Detector()

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; font-family: sans-serif; }
            body { background: #1a2634; color: white; display: flex; height: 100vh; overflow: hidden; }
            .sidebar { width: 320px; background: #243447; border-right: 1px solid #101820; display: flex; flex-direction: column; overflow-y: auto; }
            .info-card { padding: 12px; border-bottom: 1px solid #101820; text-align: center; }
            .info-label { color: #8a99a8; font-size: 10px; text-transform: uppercase; }
            .info-value { font-size: 14px; font-weight: bold; }
            .display-container { flex: 1; position: relative; }
            #map { width: 100%; height: 100%; }
            .inset-view { position: absolute; bottom: 20px; left: 20px; width: 350px; height: 200px; border: 2px solid #00aaff; border-radius: 8px; cursor: pointer; z-index: 2000; background: #000; box-shadow: 0 0 15px rgba(0,0,0,0.5); overflow: hidden; }
            .maximized { width: 100% !important; height: 100% !important; bottom: 0 !important; left: 0 !important; border: none !important; }
            #feed-img { width: 100%; height: 100%; object-fit: contain; }
            .btn { background: #00aaff; color: white; border: none; padding: 12px; width: 90%; border-radius: 4px; cursor: pointer; display: block; margin: 8px auto; text-align: center; font-size: 13px; font-weight: bold; }
            .btn-danger { background: #ff4444; }
            .btn-success { background: #00ff88; color: #1a2634; }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="info-card"><h2 style="color:#00aaff">SkySentinels Hub</h2></div>
            <div class="info-card"><div class="info-label">System Status</div><div class="info-value" id="status" style="color:#00ff88">READY</div></div>
            
            <label for="eng" class="btn">Upload Engine</label>
            <input type="file" id="eng" accept=".engine" style="display:none" onchange="upload()">
            <p id="msg" style="color:#8a99a8; font-size:11px; text-align:center; margin-bottom:10px;">Model: NONE</p>

            <div style="background: #1a2634; padding: 10px; flex-grow: 1;">
                <div class="info-label" style="text-align:center; margin-bottom:10px;">Mission Control</div>
                <button class="btn btn-success" onclick="cmd('ARM')">ARM DRONE</button>
                <button class="btn btn-danger" onclick="cmd('DISARM')">DISARM</button>
                <button class="btn" style="background:#555" onclick="cmd('PREVIEW')">PREVIEW MISSION</button>
                <button class="btn" style="background:#ffaa00" onclick="cmd('START')">START MISSION</button>
                <button class="btn" onclick="clearWaypoints()" style="background:#444">Clear Waypoints</button>
                <div class="info-card" style="border:none"><div class="info-label">Path Vertices</div><div id="wp-count" class="info-value">0 Set</div></div>
            </div>
        </div>

        <div class="display-container">
            <div id="map"></div>
            <div class="inset-view" id="video-container" onclick="toggleSize()">
                <img id="feed-img" src="/video_feed">
            </div>
        </div>

        <script>
            // Accurate CIT Chennai Coordinates
            var citCoords = [12.9150, 80.0330];
            var map = L.map('map').setView(citCoords, 18);

            // Using ESRI Satellite Imagery
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EBP, and the GIS User Community'
            }).addTo(map);

            var waypoints = [];
            var polygon = L.polygon([], {color: '#00ff88', weight: 3, fillOpacity: 0.2}).addTo(map);
            var polyline = L.polyline([], {color: '#00aaff', weight: 4, dashArray: '5, 10'}).addTo(map);

            map.on('click', function(e) {
                var marker = L.marker([e.latlng.lat, e.latlng.lng]).addTo(map);
                waypoints.push(marker);
                
                var path = waypoints.map(m => m.getLatLng());
                polyline.setLatLngs(path);
                
                // If 3 or more points, show the mission polygon
                if(path.length >= 3) {
                    polygon.setLatLngs(path);
                }
                
                document.getElementById('wp-count').innerText = waypoints.length + " Set";
            });

            function clearWaypoints() {
                waypoints.forEach(w => map.removeLayer(w));
                waypoints = [];
                polyline.setLatLngs([]);
                polygon.setLatLngs([]);
                document.getElementById('wp-count').innerText = "0 Set";
            }

            function cmd(type) {
                document.getElementById('status').innerText = type + "ING...";
                setTimeout(()=> { document.getElementById('status').innerText = type + "ED"; }, 500);
            }

            function toggleSize() { document.getElementById('video-container').classList.toggle('maximized'); }
            
            function upload() {
                let f = document.getElementById('eng').files[0];
                let fd = new FormData(); fd.append("file", f);
                document.getElementById('msg').innerText = "Switching Engine...";
                fetch('/upload', {method:'POST', body:fd}).then(r => r.json()).then(d => document.getElementById('msg').innerText = "Model: " + d.message);
            }
        </script>
    </body>
    </html>
    """

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(path)
    detector.load_engine(path)
    return jsonify(status="success", message=f"{detector.active_type.upper()}")

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            ret, frame = cam.get_frame()
            if not ret: continue
            
            frame = detector.detect(frame)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
