import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import time  # This was missing in the previous version
from ultralytics import YOLO


class YOLODetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        self.root.geometry("900x700")

        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10), padding=6)
        self.style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'))

        # Load YOLO model with GPU if available
        self.model = YOLO('yolo11s.pt') #You can change the model by changing the name here, yolo11s.pt | my_custom_model.pt | DrugsModel.pt
        self.conf_threshold = 0.25

        # Camera variables
        self.cap = None
        self.camera_mode = False
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)

        # Create GUI
        self.create_widgets()

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="YOLO Object Detection",
                  style='Title.TLabel').pack(side=tk.LEFT)

        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text=" Detection Mode ", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 15))

        self.mode_var = tk.StringVar(value="photo")

        ttk.Radiobutton(mode_frame, text="Image Mode",
                        variable=self.mode_var, value="photo",
                        command=self.mode_changed).pack(side=tk.LEFT, padx=10)

        ttk.Radiobutton(mode_frame, text="Camera Mode",
                        variable=self.mode_var, value="camera",
                        command=self.mode_changed).pack(side=tk.LEFT, padx=10)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 15))

        self.select_btn = ttk.Button(control_frame, text="Select Image",
                                     command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.camera_btn = ttk.Button(control_frame, text="Start Camera",
                                     command=self.toggle_camera, state=tk.DISABLED)
        self.camera_btn.pack(side=tk.LEFT, padx=5)

        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").pack(side=tk.LEFT, padx=(15, 5))
        self.conf_slider = tk.Scale(control_frame, from_=0.1, to=0.9, resolution=0.05,
                                    orient=tk.HORIZONTAL, command=self.update_threshold)
        self.conf_slider.set(self.conf_threshold)
        self.conf_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # Display area
        display_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(display_frame, bg='#333333', bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(main_frame, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(5, 0))

    def update_threshold(self, val):
        self.conf_threshold = float(val)
        self.status_var.set(f"Confidence threshold set to {self.conf_threshold:.2f}")

    def mode_changed(self):
        if self.mode_var.get() == "photo":
            self.select_btn.config(state=tk.NORMAL)
            self.camera_btn.config(state=tk.DISABLED, text="Start Camera")
            if self.camera_mode:
                self.stop_camera()
            self.status_var.set("Ready to process images")
        else:
            self.select_btn.config(state=tk.DISABLED)
            self.camera_btn.config(state=tk.NORMAL)
            self.status_var.set("Ready to start camera")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            self.status_var.set(f"Processing: {file_path}")
            self.root.update()
            self.process_image(file_path)

    def process_image(self, image_path):
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("Error", "Could not read the image file")
                self.status_var.set("Error loading image")
                return

            # Resize while maintaining aspect ratio
            frame = self.resize_to_canvas(frame)

            # Perform detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            annotated_frame = self.annotate_frame(frame, results)

            # Display results
            self.display_image(annotated_frame)
            self.status_var.set(f"Done processing: {image_path}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Processing error")

    def toggle_camera(self):
        if not self.camera_mode:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            self.status_var.set("Camera error")
            return

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.camera_mode = True
        self.running = True
        self.camera_btn.config(text="Stop Camera")
        self.status_var.set("Camera running - Press 'Stop Camera' to end")

        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.camera_mode = False
        self.camera_btn.config(text="Start Camera")
        self.status_var.set("Camera stopped")

    def camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("Camera error: Failed to capture frame")
                break

            # Perform detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            annotated_frame = self.annotate_frame(frame, results)

            # Display results
            self.display_image(annotated_frame)

            # Small delay to prevent GUI freeze
            time.sleep(0.03)  # Now properly imported

    def annotate_frame(self, frame, results):
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

                # Display label
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return annotated_frame

    def resize_to_canvas(self, frame):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return frame

        h, w = frame.shape[:2]
        ratio = min(canvas_width / w, canvas_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def display_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # Calculate display dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = img.size

        # Center the image
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2

        imgtk = ImageTk.PhotoImage(image=img)

        # Update display
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()