"""
Waste Classifier GUI - Simple interface with file picker
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from pathlib import Path
import json

class WasteClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Classifier - Dry Waste Detection")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        self.model = None
        self.model_path = "trained_waste_model.h5"
        self.input_shape = (224, 224, 3)
        self.classes = ['organic', 'recyclable']
        self.current_image_path = None
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        """Create the GUI layout."""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', pady=(0, 10))
        
        title_label = tk.Label(
            title_frame,
            text="♻️ Waste Classification System",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#2c3e50',
            pady=20
        )
        title_label.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Image display
        left_panel = tk.Frame(main_container, bg='white', relief='solid', borderwidth=1)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Image label
        self.image_label = tk.Label(
            left_panel,
            text="No image loaded\n\nClick 'Select Image' to begin",
            font=('Arial', 12),
            bg='white',
            fg='gray',
            width=40,
            height=20
        )
        self.image_label.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Right panel - Controls and results
        right_panel = tk.Frame(main_container, bg='#f0f0f0', width=300)
        right_panel.pack(side='right', fill='y')
        right_panel.pack_propagate(False)
        
        # Select Image Button
        select_btn = tk.Button(
            right_panel,
            text="📁 Select Image",
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            command=self.select_image,
            height=2
        )
        select_btn.pack(fill='x', pady=(0, 20))
        
        # Classify Button
        self.classify_btn = tk.Button(
            right_panel,
            text="🔍 Classify Waste",
            font=('Arial', 14, 'bold'),
            bg='#2ecc71',
            fg='white',
            activebackground='#27ae60',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            command=self.classify_image,
            height=2,
            state='disabled'
        )
        self.classify_btn.pack(fill='x', pady=(0, 20))
        
        # Results frame
        results_frame = tk.LabelFrame(
            right_panel,
            text="Classification Results",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        results_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Result labels
        self.class_label = tk.Label(
            results_frame,
            text="Class: -",
            font=('Arial', 11),
            bg='#f0f0f0',
            anchor='w'
        )
        self.class_label.pack(fill='x', padx=10, pady=5)
        
        self.confidence_label = tk.Label(
            results_frame,
            text="Confidence: -",
            font=('Arial', 11),
            bg='#f0f0f0',
            anchor='w'
        )
        self.confidence_label.pack(fill='x', padx=10, pady=5)
        
        self.dry_waste_label = tk.Label(
            results_frame,
            text="Dry Waste: -",
            font=('Arial', 11, 'bold'),
            bg='#f0f0f0',
            fg='#27ae60',
            anchor='w'
        )
        self.dry_waste_label.pack(fill='x', padx=10, pady=5)
        
        self.organic_label = tk.Label(
            results_frame,
            text="Wet Waste: -",
            font=('Arial', 11),
            bg='#f0f0f0',
            anchor='w'
        )
        self.organic_label.pack(fill='x', padx=10, pady=5)
        
        # Status text label
        self.processing_label = tk.Label(
            results_frame,
            text="",
            font=('Arial', 10, 'italic'),
            bg='#f0f0f0',
            fg='#3498db',
            anchor='center'
        )
        self.processing_label.pack(fill='x', padx=10, pady=10)
        
        # Status label
        self.status_label = tk.Label(
            right_panel,
            text="Ready",
            font=('Arial', 9),
            bg='#f0f0f0',
            fg='gray'
        )
        self.status_label.pack(side='bottom', pady=5)
        
        # Clear button only
        clear_btn = tk.Button(
            right_panel,
            text="�️ Clear Results",
            font=('Arial', 12),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            command=self.clear_results,
            height=2
        )
        clear_btn.pack(side='bottom', fill='x', pady=(0, 10))
    
    def load_model(self):
        """Load the trained model."""
        try:
            if os.path.exists(self.model_path):
                self.update_status("Loading model...")
                self.model = tf.keras.models.load_model(self.model_path)
                self.update_status("Model loaded successfully ✓")
                messagebox.showinfo("Success", f"Model loaded from {self.model_path}")
            else:
                self.update_status("No trained model found!")
                messagebox.showwarning(
                    "Model Not Found",
                    f"Model file '{self.model_path}' not found!\n\n"
                    "Please train the model first using:\n"
                    "python waste_classifier_with_kaggle.py"
                )
        except Exception as e:
            self.update_status(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
    
    def select_image(self):
        """Open file dialog to select image."""
        file_path = filedialog.askopenfilename(
            title="Select Waste Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.classify_btn.config(state='normal')
            self.update_status(f"Image loaded: {Path(file_path).name}")
            self.clear_results()
    
    def display_image(self, image_path):
        """Display selected image in the GUI."""
        try:
            # Load and resize image
            img = Image.open(image_path)
            
            # Resize to fit display (maintain aspect ratio)
            display_width = 400
            display_height = 400
            img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def classify_image(self):
        """Classify the selected image."""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        if not self.model:
            messagebox.showerror("No Model", "No trained model available!")
            return
        
        # Run classification in separate thread to keep GUI responsive
        thread = threading.Thread(target=self._classify_thread)
        thread.start()
    
    def _classify_thread(self):
        """Background thread for classification."""
        try:
            # Update UI
            self.root.after(0, lambda: self.processing_label.config(text="🔄 Processing image..."))
            self.root.after(0, lambda: self.update_status("Classifying..."))
            self.root.after(0, lambda: self.classify_btn.config(state='disabled'))
            
            # Load and preprocess image
            img = load_img(self.current_image_path, target_size=self.input_shape[:2])
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predict
            prediction = self.model.predict(img_array, verbose=0)
            
            # Calculate results
            recyclable_idx = self.classes.index('recyclable')
            organic_idx = self.classes.index('organic')
            
            recyclable_pct = prediction[0][recyclable_idx] * 100
            organic_pct = prediction[0][organic_idx] * 100
            
            predicted_class = self.classes[np.argmax(prediction[0])]
            confidence = np.max(prediction[0]) * 100
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'dry_waste_percentage': recyclable_pct,
                'organic_percentage': organic_pct
            }
            
            # Update UI with results
            self.root.after(0, lambda: self.display_results(result))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification failed:\n{e}"))
            self.root.after(0, lambda: self.update_status("Classification failed"))
            self.root.after(0, lambda: self.processing_label.config(text="❌ Classification failed"))
        
        finally:
            self.root.after(0, lambda: self.classify_btn.config(state='normal'))
    
    def display_results(self, result):
        """Display classification results."""
        # Update labels with colors
        class_text = result['predicted_class'].upper()
        if class_text == 'ORGANIC':
            class_text = 'WET WASTE'
        class_color = '#27ae60' if result['predicted_class'] == 'recyclable' else '#8b4513'
        
        self.class_label.config(
            text=f"Class: {class_text}",
            fg=class_color,
            font=('Arial', 12, 'bold')
        )
        
        self.confidence_label.config(
            text=f"Confidence: {result['confidence']:.1f}%"
        )
        
        self.dry_waste_label.config(
            text=f"♻️ Dry Waste (Recyclable): {result['dry_waste_percentage']:.1f}%",
            font=('Arial', 11, 'bold')
        )
        
        self.organic_label.config(
            text=f"🌱 Wet Waste: {result['organic_percentage']:.1f}%"
        )
        
        # Clear processing text and show completion
        self.processing_label.config(text="✅ Classification complete!")
        self.update_status("Classification complete ✓")
        
        # Store result for saving
        self.last_result = result
    

    def clear_results(self):
        """Clear classification results."""
        self.class_label.config(text="Class: -", fg='black', font=('Arial', 11))
        self.confidence_label.config(text="Confidence: -")
        self.dry_waste_label.config(text="Dry Waste: -")
        self.organic_label.config(text="Wet Waste: -")
        self.processing_label.config(text="")
        
        if hasattr(self, 'last_result'):
            delattr(self, 'last_result')
        
        self.update_status("Results cleared")
    
    def update_status(self, message):
        """Update status label."""
        self.status_label.config(text=message)

def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = WasteClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()