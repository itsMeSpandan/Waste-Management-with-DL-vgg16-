# Waste Classifier GUI Documentation

## Overview

The `waste_classifier_gui.py` file provides a user-friendly graphical interface for the waste classification system. This GUI application allows users to load images of waste items and classify them as either "organic" or "recyclable" using a pre-trained machine learning model.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dependencies](#dependencies)
- [Architecture](#architecture)
- [Class Structure](#class-structure)
- [Methods Documentation](#methods-documentation)
- [Usage](#usage)
- [Installation](#installation)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [File Structure](#file-structure)

## Features

- **Image Selection**: Browse and select waste images from the file system
- **Real-time Classification**: Classify images as organic or recyclable waste
- **Visual Display**: Display selected images with proper scaling and aspect ratio
- **Progress Tracking**: Visual progress indicator during classification
- **Results Display**: Show classification results with confidence percentages
- **Export Functionality**: Save classification results to JSON files
- **Responsive UI**: Non-blocking classification using background threads
- **Model Management**: Automatic model loading with error handling

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **GUI Framework** | Tkinter | Main GUI framework with ttk for enhanced widgets |
| **Image Processing** | PIL (Pillow) | Image loading, resizing, and display |
| **Machine Learning** | TensorFlow/Keras | Model loading and inference |
| **Array Processing** | NumPy | Numerical operations and array handling |
| **Concurrency** | Python Threading | Background processing for responsive UI |
| **File I/O** | JSON, PathLib | Result saving and file path management |

## Dependencies

```python
# Core Python Libraries
import os
import sys
import threading
import json
from pathlib import Path

# GUI Components
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import PhotoImage

# Image Processing
from PIL import Image, ImageTk

# Machine Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Numerical Computing
import numpy as np
```

## Architecture

The application follows a single-class architecture with clear separation of concerns:

```
WasteClassifierGUI
├── UI Setup (setup_ui)
├── Model Management (load_model)
├── Image Handling (select_image, display_image)
├── Classification Engine (classify_image, _classify_thread)
├── Results Management (display_results, save_results, clear_results)
└── Utility Functions (update_status)
```
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## Class Structure

### WasteClassifierGUI

The main application class that manages the entire GUI and classification workflow.

#### Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `root` | tk.Tk | Main window object |
| `model` | tf.keras.Model | Loaded machine learning model |
| `model_path` | str | Path to the trained model file |
| `input_shape` | tuple | Expected input shape (224, 224, 3) |
| `classes` | list | Classification labels ['organic', 'recyclable'] |
| `current_image_path` | str | Path to currently selected image |
| `last_result` | dict | Last classification result for saving |

#### GUI Components

The interface consists of several key components:

1. **Title Bar**: Application header with waste classification branding
2. **Left Panel**: Image display area with placeholder text
3. **Right Panel**: Controls and results section
4. **Button Controls**: Select Image, Classify Waste, Save Results, Clear
5. **Results Display**: Classification outcomes with confidence scores
6. **Progress Bar**: Visual feedback during processing
7. **Status Bar**: Current application status

## Methods Documentation

### `__init__(self, root)`

**Purpose**: Initialize the GUI application

**Parameters**:
- `root` (tk.Tk): The main tkinter window

**Functionality**:
- Sets up window properties (title, size, background)
- Initializes class attributes
- Calls UI setup and model loading methods

### `setup_ui(self)`

**Purpose**: Create and arrange all GUI components

**Layout Structure**:
```
┌─────────────────────────────────┐
│           Title Bar             │
├─────────────────┬───────────────┤
│                 │   Controls    │
│  Image Display  │   Results     │
│                 │   Status      │
└─────────────────┴───────────────┘
```
<br>
<br>

**Components Created**:
- Title frame with application branding
- Main container with left/right panels
- Image display label
- Control buttons (Select, Classify, Save, Clear)
- Results frame with classification output
- Progress bar and status label

### `load_model(self)`

**Purpose**: Load the pre-trained machine learning model

**Process**:
1. Check if model file exists at `self.model_path`
2. Load model using `tf.keras.models.load_model()`
3. Update UI status and show success/error messages
4. Handle exceptions with user-friendly error dialogs

**Error Handling**:
- File not found: Shows warning with training instructions
- Loading errors: Displays error message with exception details

### `select_image(self)`

**Purpose**: Open file dialog for image selection

**Supported Formats**:
- JPEG (*.jpg, *.jpeg)
- PNG (*.png)
- BMP (*.bmp)
- GIF (*.gif)

**Process**:
1. Open file dialog with image filters
2. Store selected file path
3. Display image in GUI
4. Enable classify button
5. Clear previous results

### `display_image(self, image_path)`

**Purpose**: Display selected image in the GUI with proper scaling

**Parameters**:
- `image_path` (str): Path to the image file

**Image Processing**:
1. Load image using PIL
2. Resize to fit display area (400x400 max) while maintaining aspect ratio
3. Convert to PhotoImage for tkinter compatibility
4. Update image label with new image

### `classify_image(self)`

**Purpose**: Initiate image classification process

**Validation**:
- Check if image is selected
- Verify model is loaded
- Start background thread for processing

### `_classify_thread(self)`

**Purpose**: Background thread for image classification to keep GUI responsive

**Classification Pipeline**:
1. **Preprocessing**:
   - Load image at target size (224, 224)
   - Convert to array format
   - Add batch dimension
   - Apply VGG16 preprocessing

2. **Inference**:
   - Run model prediction
   - Extract class probabilities

3. **Post-processing**:
   - Calculate percentages for each class
   - Determine predicted class and confidence
   - Format results dictionary

4. **UI Updates** (via `root.after()`):
   - Start/stop progress bar
   - Enable/disable buttons
   - Display results or error messages

### `display_results(self, result)`

**Purpose**: Update GUI with classification results

**Parameters**:
- `result` (dict): Classification results containing:
  - `predicted_class`: 'organic' or 'recyclable'
  - `confidence`: Overall confidence percentage
  - `dry_waste_percentage`: Recyclable percentage
  - `organic_percentage`: Organic percentage

**Visual Updates**:
- Color-coded class labels (green for recyclable, brown for organic)
- Confidence percentage display
- Individual class percentages with emojis
- Status update confirmation

### `save_results(self)`

**Purpose**: Export classification results to JSON file

**Output Format**:
```json
{
  "image_path": "/path/to/image.jpg",
  "image_name": "image.jpg",
  "predicted_class": "recyclable",
  "confidence": 87.5,
  "dry_waste_percentage": 87.5,
  "organic_percentage": 12.5
}
```

**Process**:
1. Validate results exist
2. Open save dialog
3. Create output dictionary
4. Write to JSON file with proper formatting

### `clear_results(self)`

**Purpose**: Reset all result displays to default state

**Actions**:
- Reset all result labels to "-"
- Remove color formatting
- Clear stored result data

### `update_status(self, message)`

**Purpose**: Update status bar with current application state

**Parameters**:
- `message` (str): Status message to display

## Usage

### Basic Workflow

1. **Start Application**:
   ```bash
   python waste_classifier_gui.py
   ```

2. **Load Image**:
   - Click "📁 Select Image" button
   - Browse and select a waste image
   - Image appears in the left panel

3. **Classify**:
   - Click "🔍 Classify Waste" button
   - Wait for processing (progress bar shows activity)
   - Results appear in the right panel

4. **Save Results** (Optional):
   - Click "💾 Save Results" button
   - Choose save location
   - Results exported to JSON format

5. **Clear and Repeat**:
   - Click "🗑️ Clear" to reset
   - Select new image for another classification

<br>
<br>
<br>

### Model Requirements

The application expects a trained model file at:
- **Default Path**: `trained_waste_model.h5`
- **Format**: Keras HDF5 model
- **Input Shape**: (224, 224, 3)
- **Output Classes**: ['organic', 'recyclable']
- **Preprocessing**: VGG16-style preprocessing required

## Installation

### Prerequisites

```bash
# Install required packages
pip install tensorflow
pip install pillow
pip install numpy

# Tkinter is usually included with Python
```

### Setup Steps

1. **Clone/Download**: Get the project files
2. **Install Dependencies**: Run pip install commands
3. **Train Model**: Ensure `trained_waste_model.h5` exists
4. **Run Application**: Execute the GUI script

## Configuration

### Customizable Settings

| Setting | Location | Purpose |
|---------|----------|---------|
| `model_path` | `__init__()` | Path to trained model |
| `input_shape` | `__init__()` | Expected input dimensions |
| `classes` | `__init__()` | Classification labels |
| Window Size | `__init__()` | GUI dimensions (800x600) |
| Display Size | `display_image()` | Image display size (400x400) |

<br>
<br>
<br>
<br>
<br>

### Model Path Configuration

To use a different model file:

```python
# In __init__ method
self.model_path = "path/to/your/model.h5"
```

## Error Handling

The application includes comprehensive error handling:

### Model Loading Errors
- **File Not Found**: Warning dialog with training instructions
- **Loading Exception**: Error dialog with technical details

### Image Processing Errors
- **Invalid Format**: Error message for unsupported files
- **Corrupted Files**: Graceful handling with user notification

### Classification Errors
- **Model Issues**: Error dialog during prediction
- **Threading Errors**: Proper cleanup and UI reset

### File I/O Errors
- **Save Failures**: Error messages for write permissions
- **Path Issues**: Validation and user feedback

## File Structure

```
waste_classifier_gui.py          # Main GUI application
├── WasteClassifierGUI class     # Main application class
│   ├── __init__()              # Initialization
│   ├── setup_ui()              # GUI layout
│   ├── load_model()            # Model management
│   ├── select_image()          # File selection
│   ├── display_image()         # Image display
│   ├── classify_image()        # Classification trigger
│   ├── _classify_thread()      # Background processing
│   ├── display_results()       # Results display
│   ├── save_results()          # Export functionality
│   ├── clear_results()         # UI reset
│   └── update_status()         # Status updates
└── main()                      # Entry point
```

## Performance Considerations

### Memory Management
- Images are resized for display to save memory
- Model loaded once at startup
- Proper cleanup of PIL objects

### Threading
- Classification runs in background thread
- UI remains responsive during processing
- Proper thread synchronization with `root.after()`

### Optimization Opportunities
- Model caching for faster subsequent predictions
- Batch processing for multiple images
- GPU acceleration if available

## Future Enhancements

### Potential Features
- **Batch Processing**: Multiple image classification
- **Real-time Camera**: Live camera feed integration
- **History**: Classification history with thumbnails
- **Model Selection**: Choose between different models
- **Advanced Metrics**: Detailed analysis and statistics
- **Export Options**: Multiple output formats (CSV, XML)

### Technical Improvements
- **Drag & Drop**: Image drag-and-drop support
- **Keyboard Shortcuts**: Hotkeys for common actions
- **Themes**: Dark/light mode support
- **Internationalization**: Multi-language support
- **Logging**: Detailed application logging

---

*This documentation covers all aspects of the Waste Classifier GUI application. For technical support or feature requests, please refer to the project repository or contact the development team.*