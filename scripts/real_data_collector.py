# ==================== scripts/real_data_collector.py ====================
import cv2
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk

class RealDataCollector:
    def __init__(self, output_dir="./data/real_data/"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.annotations_file = os.path.join(output_dir, "annotations.json")
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Load existing annotations
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = []
        
        self.damage_types = ["missing", "cracked", "broken", "loose", "scratched"]
        self.part_types = [
            "seat", "back", "front_left_leg", "front_right_leg", 
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
        ]
    
    def collect_image(self):
        """Collect a single image with annotations"""
        
        # Capture image
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Press SPACE to capture image, ESC to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Capture Image', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"real_{timestamp}.jpg"
                filepath = os.path.join(self.images_dir, filename)
                
                cv2.imwrite(filepath, frame)
                print(f"Image saved: {filepath}")
                
                # Get annotations
                self.annotate_image(filename, frame)
                break
            elif key == 27:  # ESC to quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def annotate_image(self, filename, image):
        """Annotate captured image"""
        
        # Create annotation GUI
        root = tk.Tk()
        root.title("Image Annotation")
        
        # Display image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((400, 300))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        label = tk.Label(root, image=img_tk)
        label.pack()
        
        # Annotation data
        damages = []
        
        def add_damage():
            # Select damage type
            damage_type = simpledialog.askstring("Damage Type", 
                                                f"Enter damage type: {', '.join(self.damage_types)}")
            if damage_type not in self.damage_types:
                messagebox.showerror("Error", "Invalid damage type")
                return
            
            # Select part
            part = simpledialog.askstring("Part", 
                                        f"Enter part name: {', '.join(self.part_types)}")
            if part not in self.part_types:
                messagebox.showerror("Error", "Invalid part name")
                return
            
            # Add damage
            damages.append({
                "type": damage_type,
                "part": part,
                "severity": 0.8  # Default severity
            })
            
            messagebox.showinfo("Success", f"Added {damage_type} damage to {part}")
        
        def finish_annotation():
            if not damages:
                messagebox.showwarning("Warning", "No damage annotated!")
                return
            
            # Create annotation
            annotation = {
                "image_id": len(self.annotations),
                "filename": filename,
                "width": image.shape[1],
                "height": image.shape[0],
                "damages": damages,
                "timestamp": datetime.now().isoformat()
            }
            
            self.annotations.append(annotation)
            
            # Save annotations
            with open(self.annotations_file, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            
            messagebox.showinfo("Success", "Annotation saved!")
            root.destroy()
        
        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Add Damage", command=add_damage).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Finish", command=finish_annotation).pack(side=tk.LEFT, padx=5)
        
        root.mainloop()
    
    def batch_collect(self, num_images=10):
        """Collect multiple images"""
        for i in range(num_images):
            print(f"Collecting image {i+1}/{num_images}")
            self.collect_image()
            
            if i < num_images - 1:
                input("Press Enter to continue to next image...")