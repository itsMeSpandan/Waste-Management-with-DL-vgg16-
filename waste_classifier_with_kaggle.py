"""
Dry Waste Classifier with Kaggle Dataset Integration
This script automatically downloads waste classification data from Kaggle,
trains a VGG16 model, and classifies waste images.
"""

import os
import sys
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Tuple, List
import json

class WasteClassifierKaggle:
    """Complete waste classifier with Kaggle dataset integration."""
    
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.classes = ['organic', 'recyclable']  # Binary classification
        self.dataset_path = "waste_dataset"
        self.model_path = "trained_waste_model.h5"
        
    def setup_kaggle_credentials(self):
        """Setup Kaggle API credentials."""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if kaggle_json.exists():
            print("✓ Kaggle credentials found")
            return True
        
        print("\n" + "="*60)
        print("KAGGLE API SETUP REQUIRED")
        print("="*60)
        print("\nTo download datasets from Kaggle:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. This downloads kaggle.json")
        print(f"5. Place kaggle.json in: {kaggle_dir}")
        print("\nAlternatively, enter your Kaggle credentials now:")
        
        username = input("\nKaggle Username (or 'skip'): ").strip()
        if username.lower() == 'skip':
            return False
        
        key = input("Kaggle API Key: ").strip()
        
        # Create kaggle directory
        kaggle_dir.mkdir(exist_ok=True, parents=True)
        
        # Create kaggle.json
        kaggle_config = {
            "username": username,
            "key": key
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(kaggle_config, f)
        
        # Set permissions (important for Linux/Mac)
        try:
            os.chmod(kaggle_json, 0o600)
        except:
            pass
        
        print(f"✓ Kaggle credentials saved to {kaggle_json}")
        return True
    
    def download_kaggle_dataset(self):
        """Download waste classification dataset from Kaggle."""
        print("\n" + "="*60)
        print("DOWNLOADING KAGGLE DATASET")
        print("="*60)
        
        # Try to import kaggle
        try:
            import kaggle
        except ImportError:
            print("Installing Kaggle API...")
            os.system(f"{sys.executable} -m pip install kaggle")
            import kaggle
        
        # Setup credentials
        if not self.setup_kaggle_credentials():
            print("\n⚠ Kaggle setup skipped. Cannot download dataset.")
            return False
        
        # Create dataset directory
        dataset_path = Path(self.dataset_path)
        if dataset_path.exists():
            print(f"✓ Dataset directory already exists: {dataset_path}")
            # Check if data already downloaded
            subdirs = list(dataset_path.glob('*/'))
            if subdirs:
                print(f"✓ Found existing data: {[d.name for d in subdirs[:5]]}")
                return True
        
        dataset_path.mkdir(exist_ok=True)
        
        # Popular waste classification datasets
        datasets = [
            "techsash/waste-classification-data",
            "mostafaabla/garbage-classification",
            "asdasdasasdas/garbage-classification"
        ]
        
        print("\nAvailable datasets:")
        for i, ds in enumerate(datasets, 1):
            print(f"{i}. {ds}")
        
        choice = input(f"\nSelect dataset (1-{len(datasets)}) or press Enter for default: ").strip()
        
        if not choice:
            choice = "1"
        
        try:
            dataset_name = datasets[int(choice) - 1]
        except:
            dataset_name = datasets[0]
        
        print(f"\nDownloading: {dataset_name}")
        print("This may take a few minutes...")
        
        try:
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(dataset_path),
                unzip=True
            )
            print("✓ Dataset downloaded successfully!")
            return True
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def organize_dataset(self):
        """Organize downloaded dataset into train/val structure."""
        print("\n" + "="*60)
        print("ORGANIZING DATASET")
        print("="*60)
        
        dataset_root = Path(self.dataset_path)
        
        # Find all directories with images
        all_dirs = [d for d in dataset_root.rglob('*') if d.is_dir()]
        
        # Category mapping for different dataset structures
        organic_keywords = ['organic', 'compost', 'biological', 'food', 'o']
        recyclable_keywords = ['recyclable', 'recycle', 'plastic', 'paper', 
                              'metal', 'glass', 'cardboard', 'r']
        
        # Prepare organized structure
        train_dir = dataset_root / 'organized' / 'train'
        val_dir = dataset_root / 'organized' / 'val'
        
        for split_dir in [train_dir, val_dir]:
            for class_name in self.classes:
                (split_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        image_count = {'organic': 0, 'recyclable': 0}
        
        # Process each directory
        for dir_path in all_dirs:
            if 'organized' in str(dir_path):
                continue
            
            dir_name = dir_path.name.lower()
            
            # Determine category
            category = None
            if any(keyword in dir_name for keyword in organic_keywords):
                category = 'organic'
            elif any(keyword in dir_name for keyword in recyclable_keywords):
                category = 'recyclable'
            else:
                continue  # Skip unknown categories
            
            # Find all images in directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(dir_path.glob(ext))
                image_files.extend(dir_path.glob(ext.upper()))
            
            if not image_files:
                continue
            
            # Split into train/val
            train_files, val_files = train_test_split(
                image_files, test_size=0.2, random_state=42
            )
            
            # Copy files
            for file_list, target_base in [(train_files, train_dir), (val_files, val_dir)]:
                target_class_dir = target_base / category
                
                for img_file in file_list:
                    target_file = target_class_dir / f"{dir_name}_{img_file.name}"
                    if not target_file.exists():
                        shutil.copy2(img_file, target_file)
                        image_count[category] += 1
        
        print(f"\n✓ Dataset organized:")
        print(f"  - Organic: {image_count['organic']} images")
        print(f"  - Recyclable: {image_count['recyclable']} images")
        
        # Check if we have enough data
        if image_count['organic'] == 0 or image_count['recyclable'] == 0:
            print("\n⚠ Warning: Some categories have no images!")
            print("The dataset structure might be different. Attempting alternative organization...")
            return self.organize_dataset_alternative()
        
        return True
    
    def organize_dataset_alternative(self):
        """Alternative organization for different dataset structures."""
        dataset_root = Path(self.dataset_path)
        
        # Look for TRAIN/TEST structure
        train_test_dirs = []
        for possible_name in ['TRAIN', 'TEST', 'train', 'test', 'training', 'testing']:
            possible_dir = dataset_root / possible_name
            if possible_dir.exists():
                train_test_dirs.append(possible_dir)
        
        if train_test_dirs:
            print(f"Found structured directories: {[d.name for d in train_test_dirs]}")
            
            # Create organized structure
            train_dir = dataset_root / 'organized' / 'train'
            val_dir = dataset_root / 'organized' / 'val'
            
            for split_dir in [train_dir, val_dir]:
                for class_name in self.classes:
                    (split_dir / class_name).mkdir(parents=True, exist_ok=True)
            
            # Use first dir as training source
            if train_test_dirs:
                source_dir = train_test_dirs[0]
                
                # Map O->organic, R->recyclable
                class_mapping = {'o': 'organic', 'r': 'recyclable'}
                
                for subdir in source_dir.iterdir():
                    if not subdir.is_dir():
                        continue
                    
                    dir_key = subdir.name.lower()[0]  # First letter
                    if dir_key in class_mapping:
                        category = class_mapping[dir_key]
                        
                        # Get images
                        image_files = []
                        for ext in ['*.jpg', '*.jpeg', '*.png']:
                            image_files.extend(subdir.rglob(ext))
                        
                        # Split and copy
                        train_files, val_files = train_test_split(
                            image_files, test_size=0.2, random_state=42
                        )
                        
                        for file_list, target_base in [(train_files, train_dir), (val_files, val_dir)]:
                            for img_file in file_list:
                                target = target_base / category / img_file.name
                                if not target.exists():
                                    shutil.copy2(img_file, target)
                
                print("✓ Alternative organization completed")
                return True
        
        print("✗ Could not organize dataset automatically")
        return False
    
    def build_model(self):
        """Build VGG16-based waste classification model."""
        print("\n" + "="*60)
        print("BUILDING MODEL")
        print("="*60)
        
        # Load pre-trained VGG16
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(self.classes), activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✓ Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train_model(self, epochs=15, batch_size=32):
        """Train the model on Kaggle dataset."""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        if not self.model:
            self.build_model()
        
        train_dir = Path(self.dataset_path) / 'organized' / 'train'
        val_dir = Path(self.dataset_path) / 'organized' / 'val'
        
        if not train_dir.exists():
            print("✗ Training data not found. Run dataset download first.")
            return None
        
        # Data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.15,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.classes
        )
        
        val_generator = val_datagen.flow_from_directory(
            str(val_dir),
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.classes
        )
        
        print(f"\n✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=2,
                verbose=1
            )
        ]
        
        # Train
        print(f"\nTraining for {epochs} epochs...")
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        print(f"\n✓ Training completed!")
        print(f"✓ Best model saved to: {self.model_path}")
        
        # Plot training history
        self.plot_history(history)
        
        return history
    
    def plot_history(self, history):
        """Plot training history with enhanced visualization."""
        print("\n📊 Generating training plots...")
        
        # Normalize history to a simple dict of lists
        hist = history.history if hasattr(history, 'history') else history
        if not isinstance(hist, dict):
            print("❌ Unexpected history format; cannot plot.")
            return

        # Utility: pick best-matching keys for metrics
        def pick_key(candidates, default=None):
            for k in candidates:
                if k in hist:
                    return k
            return default

        # Resolve accuracy keys robustly
        acc_key = pick_key([
            'accuracy',
            'acc',
            'categorical_accuracy',
            'binary_accuracy',
            'sparse_categorical_accuracy'
        ])
        val_acc_key = pick_key([
            'val_accuracy',
            'val_acc',
            f"val_{acc_key}" if acc_key else None
        ])

        # Resolve loss keys
        loss_key = pick_key(['loss']) or next((k for k in hist.keys() if k.endswith('loss') and not k.startswith('val')), None)
        val_loss_key = pick_key(['val_loss', f"val_{loss_key}" if loss_key else None]) or next((k for k in hist.keys() if k.startswith('val') and k.endswith('loss')), None)

        # Resolve learning rate keys (optional)
        lr_key = pick_key(['lr', 'learning_rate'])

        # Extract series (default to empty list if missing)
        acc = hist.get(acc_key, []) if acc_key else []
        val_acc = hist.get(val_acc_key, []) if val_acc_key else []
        loss = hist.get(loss_key, []) if loss_key else []
        val_loss = hist.get(val_loss_key, []) if val_loss_key else []
        lrs = hist.get(lr_key, []) if lr_key else []

        # Debug: print detected keys and lengths to help diagnose blank plots
        print("Detected history keys:", list(hist.keys()))
        print(f"Using keys -> acc: {acc_key}, val_acc: {val_acc_key}, loss: {loss_key}, val_loss: {val_loss_key}, lr: {lr_key}")
        print(f"Lengths -> acc: {len(acc)}, val_acc: {len(val_acc)}, loss: {len(loss)}, val_loss: {len(val_loss)}, lr: {len(lrs)}")

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Waste Classifier Training Results', fontsize=16, fontweight='bold')

        # Plot 1: Accuracy
        if len(acc) or len(val_acc):
            if len(acc):
                axes[0, 0].plot(acc, 'b-', label='Training Accuracy', linewidth=2)
            if len(val_acc):
                axes[0, 0].plot(val_acc, 'r-', label='Validation Accuracy', linewidth=2)
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, 'No Accuracy Metrics Found', ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12, alpha=0.7)
        axes[0, 0].set_title('Model Accuracy Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Loss
        if len(loss) or len(val_loss):
            if len(loss):
                axes[0, 1].plot(loss, 'b-', label='Training Loss', linewidth=2)
            if len(val_loss):
                axes[0, 1].plot(val_loss, 'r-', label='Validation Loss', linewidth=2)
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, 'No Loss Metrics Found', ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12, alpha=0.7)
        axes[0, 1].set_title('Model Loss Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Learning Rate (if available)
        if len(lrs):
            axes[1, 0].plot(lrs, 'g-', linewidth=2)
            axes[1, 0].set_yscale('log')
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                            ha='center', va='center', transform=axes[1, 0].transAxes,
                            fontsize=12, alpha=0.7)
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Training Summary
        def safe_last(values):
            try:
                return float(values[-1]) if len(values) else None
            except Exception:
                return None

        final_train_acc = safe_last(acc)
        final_val_acc = safe_last(val_acc)
        best_val_acc = float(max(val_acc)) if len(val_acc) else None

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "N/A"

        epochs_len = max(len(acc), len(val_acc), len(loss), len(val_loss))
        summary_text = f"""Training Summary:
        
Final Training Accuracy: {fmt(final_train_acc)}
Final Validation Accuracy: {fmt(final_val_acc)}
Best Validation Accuracy: {fmt(best_val_acc)}

Total Epochs: {epochs_len}
Classes: {', '.join(self.classes)}"""

        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        axes[1, 1].set_title('Training Summary', fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history saved to training_history.png")
        plt.show()

        # Save standardized training history data
        history_data = {
            'accuracy': list(map(float, acc)) if len(acc) else [],
            'val_accuracy': list(map(float, val_acc)) if len(val_acc) else [],
            'loss': list(map(float, loss)) if len(loss) else [],
            'val_loss': list(map(float, val_loss)) if len(val_loss) else [],
            'lr': list(map(float, lrs)) if len(lrs) else [],
            'epochs': epochs_len,
            'final_metrics': {
                'train_accuracy': float(final_train_acc) if final_train_acc is not None else None,
                'val_accuracy': float(final_val_acc) if final_val_acc is not None else None,
                'best_val_accuracy': float(best_val_acc) if best_val_acc is not None else None
            }
        }

        with open('training_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
        print("✓ Training data saved to training_history.json")
    
    def show_training_plot(self):
        """Display saved training history plot."""
        print("\n📊 Loading training history...")
        
        # Try to load training history data
        if os.path.exists('training_history.json'):
            with open('training_history.json', 'r') as f:
                history_data = json.load(f)
            
            # Create mock history object for plotting
            class MockHistory:
                def __init__(self, data):
                    self.history = data
            
            # Warn if saved arrays are empty
            try:
                acc_len = len(history_data.get('accuracy', []))
                val_acc_len = len(history_data.get('val_accuracy', []))
                loss_len = len(history_data.get('loss', []))
                val_loss_len = len(history_data.get('val_loss', []))
                print(f"Saved history lengths -> acc: {acc_len}, val_acc: {val_acc_len}, loss: {loss_len}, val_loss: {val_loss_len}")
                if acc_len == 0 and val_acc_len == 0 and loss_len == 0 and val_loss_len == 0:
                    print("⚠ Saved history arrays are empty. The previous training may not have run, or the file is stale.")
                    print("Tip: Retrain the model or delete training_history.json and run training again.")
            except Exception:
                pass

            mock_history = MockHistory(history_data)
            self.plot_history(mock_history)
            return True
        
        # Fallback: try to show saved PNG
        elif os.path.exists('training_history.png'):
            print("✓ Displaying saved training plot...")
            import matplotlib.image as mpimg
            
            img = mpimg.imread('training_history.png')
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Saved Training History', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()
            return True
        
        else:
            print("❌ No training history found!")
            print("Train a model first to generate training plots.")
            return False
    
    def load_trained_model(self):
        """Load previously trained model."""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✓ Loaded trained model from {self.model_path}")
            return True
        return False
    
    def classify_image(self, image_path):
        """Classify a single image."""
        if not self.model:
            if not self.load_trained_model():
                print("✗ No trained model available!")
                return None
        
        # Load and preprocess image
        img = load_img(image_path, target_size=self.input_shape[:2])
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict
        prediction = self.model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        predicted_class = self.classes[class_idx]
        
        # Calculate dry waste percentage
        recyclable_percentage = prediction[0][self.classes.index('recyclable')] * 100
        
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'dry_waste_percentage': float(recyclable_percentage),
            'is_dry_waste': predicted_class == 'recyclable',
            'all_probabilities': {
                self.classes[i]: float(prediction[0][i] * 100) 
                for i in range(len(self.classes))
            }
        }
        
        return result
    
    def analyze_image_with_visualization(self, image_path):
        """Analyze and visualize image classification."""
        result = self.classify_image(image_path)
        
        if not result:
            return
        
        # Display results
        print("\n" + "="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Image: {image_path}")
        print(f"Class: {result['predicted_class'].upper()}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Dry Waste (Recyclable): {result['dry_waste_percentage']:.2f}%")
        print(f"Organic: {result['all_probabilities']['organic']:.2f}%")
        
        # Visualize
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Image: {Path(image_path).name}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        classes = list(result['all_probabilities'].keys())
        percentages = list(result['all_probabilities'].values())
        colors = ['green' if c == 'recyclable' else 'brown' for c in classes]
        
        plt.bar(classes, percentages, color=colors, alpha=0.7)
        plt.ylabel('Percentage (%)')
        plt.title('Classification Results')
        plt.ylim(0, 100)
        
        for i, (cls, pct) in enumerate(zip(classes, percentages)):
            plt.text(i, pct + 3, f'{pct:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return result

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("WASTE CLASSIFIER WITH KAGGLE DATA")
    print("="*60)
    
    classifier = WasteClassifierKaggle()
    
    print("\nWhat would you like to do?")
    print("1. Download dataset and train model")
    print("2. Train with existing downloaded dataset")  
    print("3. Classify an image (requires trained model)")
    print("4. 📊 Show training graphs/plots")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        # Download and train
        if classifier.download_kaggle_dataset():
            classifier.organize_dataset()
            
            epochs = input("\nNumber of training epochs (default 15): ").strip()
            epochs = int(epochs) if epochs else 15
            
            classifier.train_model(epochs=epochs)
            
            test_image = input("\nTest with an image? Enter path (or skip): ").strip()
            if test_image and os.path.exists(test_image):
                classifier.analyze_image_with_visualization(test_image)
    
    elif choice == "2":
        # Train with existing data
        if Path(classifier.dataset_path).exists():
            if not (Path(classifier.dataset_path) / 'organized').exists():
                classifier.organize_dataset()
            
            epochs = input("\nNumber of training epochs (default 15): ").strip()
            epochs = int(epochs) if epochs else 15
            
            classifier.train_model(epochs=epochs)
        else:
            print("✗ Dataset not found. Please download first (option 1)")
    
    elif choice == "3":
        # Classify image
        image_path = input("\nEnter image path: ").strip()
        
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return
        
        classifier.analyze_image_with_visualization(image_path)
        
        # Save results
        result = classifier.classify_image(image_path)
        with open('classification_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\n✓ Results saved to classification_result.json")
    
    elif choice == "4":
        # Show training plots
        print("\n📊 TRAINING VISUALIZATION")
        print("="*60)
        classifier.show_training_plot()
    
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()