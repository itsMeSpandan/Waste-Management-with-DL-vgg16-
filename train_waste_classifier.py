import os
import json
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

class WasteDatasetTrainer:
    """
    Enhanced waste classifier that can download and train on Kaggle datasets.
    """
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
        self.model = None
        self.input_shape = (224, 224, 3)
        self.classes = ['organic_waste', 'dry_waste', 'mixed_waste']
        self.history = None
        
    def setup_kaggle_api(self):
        """
        Setup Kaggle API for dataset download.
        Requires kaggle.json in ~/.kaggle/ directory.
        """
        try:
            import kaggle
            print("Kaggle API configured successfully!")
            return True
        except Exception as e:
            print(f"Kaggle API setup failed: {e}")
            print("\nTo setup Kaggle API:")
            print("1. Go to https://www.kaggle.com/settings")
            print("2. Create New API Token (downloads kaggle.json)")
            print("3. Place kaggle.json in ~/.kaggle/ directory")
            print("4. Run: pip install kaggle")
            return False
    
    def download_waste_dataset(self, dataset_name: str = "techsash/waste-classification-data"):
        """
        Download waste classification dataset from Kaggle.
        
        Args:
            dataset_name: Kaggle dataset identifier
        """
        if not self.setup_kaggle_api():
            return False
            
        try:
            import kaggle
            
            # Create dataset directory
            dataset_dir = Path("waste_dataset")
            dataset_dir.mkdir(exist_ok=True)
            
            print(f"Downloading dataset: {dataset_name}")
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(dataset_dir), 
                unzip=True
            )
            
            self.dataset_path = str(dataset_dir)
            print(f"Dataset downloaded to: {self.dataset_path}")
            return True
            
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return False
    
    def prepare_dataset_structure(self):
        """
        Organize downloaded dataset into train/validation structure.
        """
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            print("Dataset path not found. Please download dataset first.")
            return False
        
        # Common waste dataset structures to handle
        possible_structures = [
            # Structure 1: Direct class folders
            ['O', 'R'],  # Organic, Recyclable (dry waste)
            ['organic', 'recyclable'],
            ['compost', 'recycle', 'trash'],
            # Structure 2: TRAIN/TEST folders with classes
            ['TRAIN', 'TEST'],
        ]
        
        dataset_root = Path(self.dataset_path)
        print(f"Exploring dataset structure in: {dataset_root}")
        
        # List all subdirectories
        subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
        print(f"Found directories: {[d.name for d in subdirs]}")
        
        # Try to identify the structure
        train_dir = dataset_root / "prepared_data" / "train"
        val_dir = dataset_root / "prepared_data" / "val"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        for class_name in self.classes:
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)
        
        print("Dataset structure prepared!")
        return True
    
    def create_dataset_from_folder(self, source_folder: str):
        """
        Create organized dataset from a folder structure.
        Maps common waste categories to our classes.
        """
        source_path = Path(source_folder)
        if not source_path.exists():
            print(f"Source folder not found: {source_folder}")
            return False
        
        # Mapping from common dataset categories to our classes
        category_mapping = {
            # Organic waste variants
            'organic': 'organic_waste',
            'compost': 'organic_waste', 
            'biological': 'organic_waste',
            'food': 'organic_waste',
            'O': 'organic_waste',
            
            # Dry waste variants
            'recyclable': 'dry_waste',
            'recycle': 'dry_waste',
            'paper': 'dry_waste',
            'plastic': 'dry_waste',
            'metal': 'dry_waste',
            'glass': 'dry_waste',
            'cardboard': 'dry_waste',
            'R': 'dry_waste',
            
            # Mixed/other waste
            'trash': 'mixed_waste',
            'waste': 'mixed_waste',
            'non-recyclable': 'mixed_waste',
        }
        
        train_dir = Path(self.dataset_path) / "prepared_data" / "train"
        val_dir = Path(self.dataset_path) / "prepared_data" / "val"
        
        # Process each category folder
        for category_folder in source_path.iterdir():
            if not category_folder.is_dir():
                continue
                
            category_name = category_folder.name.lower()
            mapped_class = None
            
            # Find matching class
            for key, value in category_mapping.items():
                if key in category_name:
                    mapped_class = value
                    break
            
            if not mapped_class:
                print(f"Unmapped category: {category_name}, assigning to mixed_waste")
                mapped_class = 'mixed_waste'
            
            print(f"Mapping {category_name} -> {mapped_class}")
            
            # Get all images in category
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(category_folder.glob(ext))
                image_files.extend(category_folder.glob(ext.upper()))
            
            if not image_files:
                print(f"No images found in {category_name}")
                continue
            
            # Split into train/val (80/20)
            train_files, val_files = train_test_split(
                image_files, test_size=0.2, random_state=42
            )
            
            # Copy files to organized structure
            import shutil
            
            for file_list, target_dir in [(train_files, train_dir), (val_files, val_dir)]:
                target_class_dir = target_dir / mapped_class
                
                for img_file in file_list:
                    target_file = target_class_dir / f"{category_name}_{img_file.name}"
                    shutil.copy2(img_file, target_file)
            
            print(f"Processed {len(train_files)} train, {len(val_files)} val images for {mapped_class}")
        
        return True
    
    def build_model(self):
        """Build VGG16-based model for waste classification."""
        # Load pre-trained VGG16
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(self.classes), activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        return self.model
    
    def setup_data_generators(self, batch_size: int = 32):
        """Setup data generators for training."""
        train_dir = Path(self.dataset_path) / "prepared_data" / "train"
        val_dir = Path(self.dataset_path) / "prepared_data" / "val"
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
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
        
        return train_generator, val_generator
    
    def train_model(self, epochs: int = 20, batch_size: int = 32):
        """Train the model on waste classification data."""
        if not self.model:
            self.build_model()
        
        # Setup data generators
        train_gen, val_gen = self.setup_data_generators(batch_size)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'best_waste_model.h5',
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
                factor=0.2,
                patience=3,
                verbose=1
            )
        ]
        
        # Train model
        print(f"Starting training for {epochs} epochs...")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.history:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function for training waste classifier."""
    trainer = WasteDatasetTrainer()
    
    print("Waste Classification Model Trainer")
    print("=" * 40)
    
    choice = input("""
Choose an option:
1. Download dataset from Kaggle and train
2. Use existing local dataset
3. Exit

Enter choice (1-3): """).strip()
    
    if choice == "1":
        # Download from Kaggle
        dataset_options = [
            "techsash/waste-classification-data",
            "asdasdasasdas/garbage-classification",
            "mostafaabla/garbage-classification"
        ]
        
        print("\nAvailable datasets:")
        for i, dataset in enumerate(dataset_options, 1):
            print(f"{i}. {dataset}")
        
        dataset_choice = input(f"\nEnter dataset number (1-{len(dataset_options)}): ").strip()
        
        try:
            dataset_idx = int(dataset_choice) - 1
            selected_dataset = dataset_options[dataset_idx]
            
            if trainer.download_waste_dataset(selected_dataset):
                trainer.prepare_dataset_structure()
                
                # Look for dataset folders to organize
                dataset_root = Path(trainer.dataset_path)
                possible_data_dirs = []
                for item in dataset_root.iterdir():
                    if item.is_dir() and item.name not in ['prepared_data']:
                        possible_data_dirs.append(item)
                
                if possible_data_dirs:
                    print(f"\nFound data directories: {[d.name for d in possible_data_dirs]}")
                    data_dir = possible_data_dirs[0]  # Use first one
                    trainer.create_dataset_from_folder(str(data_dir))
                
        except (ValueError, IndexError):
            print("Invalid choice")
            return
    
    elif choice == "2":
        # Use local dataset
        dataset_path = input("Enter path to dataset folder: ").strip()
        if not os.path.exists(dataset_path):
            print("Dataset path not found")
            return
        
        trainer.dataset_path = dataset_path
        trainer.prepare_dataset_structure()
        trainer.create_dataset_from_folder(dataset_path)
    
    else:
        return
    
    # Train the model
    epochs = int(input("\nEnter number of epochs (default 20): ") or "20")
    batch_size = int(input("Enter batch size (default 32): ") or "32")
    
    print("\nStarting training...")
    trainer.train_model(epochs=epochs, batch_size=batch_size)
    
    # Plot results
    trainer.plot_training_history()
    
    # Save final model
    if trainer.model:
        trainer.model.save("trained_waste_classifier.h5")
        print("\nTrained model saved as 'trained_waste_classifier.h5'")
        print("You can now use this model with the main classifier script!")

if __name__ == "__main__":
    main()