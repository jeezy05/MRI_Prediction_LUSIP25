"""
Gradio app for MRI Data Preprocessing Pipeline
Supports bias field correction, head mask creation, and image visualization
"""

import os
import tempfile
import gradio as gr
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from PIL import Image
import traceback
from pathlib import Path

# ==================== Processing Functions ====================

def preprocess_mri(
    uploaded_file,
    bias_correction: bool = True,
    mask_creation: bool = True,
    intensity_normalization: bool = True,
    slice_to_display: int = None
):
    """
    Main preprocessing pipeline for MRI data
    """
    try:
        # Read the uploaded file
        if uploaded_file is None:
            return None, None, None, "Error: Please upload an MRI image file"
        
        # Get file path
        file_path = uploaded_file.name
        
        # Read with SimpleITK for detailed processing
        raw_img_sitk = sitk.ReadImage(file_path, sitk.sitkFloat32)
        raw_img_sitk = sitk.DICOMOrient(raw_img_sitk, 'RPS')
        raw_img_arr = sitk.GetArrayFromImage(raw_img_sitk)
        
        # Set default slice if not provided
        if slice_to_display is None:
            slice_to_display = raw_img_arr.shape[0] // 2
        
        # Ensure slice is within bounds
        slice_to_display = min(slice_to_display, raw_img_arr.shape[0] - 1)
        
        processing_log = ["Starting MRI preprocessing pipeline..."]
        processed_img = raw_img_sitk
        
        # Step 1: Intensity Normalization
        if intensity_normalization:
            processed_img = sitk.RescaleIntensity(processed_img, 0, 255)
            processing_log.append("✓ Intensity normalization applied")
        
        # Step 2: Head mask creation
        if mask_creation:
            transformed = sitk.RescaleIntensity(processed_img, 0, 255)
            head_mask = sitk.LiThreshold(transformed, 0, 1)
            processing_log.append("✓ Head mask created using Li thresholding")
        
        # Step 3: Bias field correction
        if bias_correction:
            try:
                shrink_factor_bias = 4
                inputImage = sitk.Shrink(processed_img, [shrink_factor_bias] * processed_img.GetDimension())
                maskImage = sitk.Shrink(head_mask, [shrink_factor_bias] * processed_img.GetDimension())
                
                bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrected = bias_corrector.Execute(inputImage, maskImage)
                
                log_bias_field = bias_corrector.GetLogBiasFieldAsImage(processed_img)
                corrected_image = processed_img / sitk.Exp(log_bias_field)
                processed_img = corrected_image
                processing_log.append("✓ Bias field correction applied (N4)")
            except Exception as e:
                processing_log.append(f"⚠ Bias correction skipped: {str(e)}")
        
        # Prepare output visualization
        processed_arr = sitk.GetArrayFromImage(processed_img)
        
        # Create comparison visualization
        output_img = create_slice_visualization(raw_img_arr, processed_arr, slice_to_display)
        
        # Log message
        log_text = "\n".join(processing_log)
        
        return output_img, log_text, processed_arr, "Processing completed successfully!"
    
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, error_msg


def create_slice_visualization(before, after, slice_idx):
    """Create a side-by-side comparison of before and after slices"""
    fig = plt.figure(figsize=(12, 5))
    
    # Ensure slice index is valid
    slice_idx = min(slice_idx, before.shape[0] - 1)
    
    ax1 = fig.add_subplot(121)
    ax1.imshow(before[slice_idx, :, :], cmap='gray')
    ax1.set_title(f'Original (Slice {slice_idx})', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(after[slice_idx, :, :], cmap='gray')
    ax2.set_title(f'Processed (Slice {slice_idx})', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def save_processed_image(processed_array, original_path):
    """Save the processed image as NIfTI file"""
    try:
        if processed_array is None:
            return None, "No processed image to save"
        
        # Read original to get metadata
        original_img = sitk.ReadImage(original_path, sitk.sitkFloat32)
        
        # Create new image from processed array
        processed_img = sitk.GetImageFromArray(processed_array)
        
        # Copy metadata
        processed_img.SetSpacing(original_img.GetSpacing())
        processed_img.SetOrigin(original_img.GetOrigin())
        processed_img.SetDirection(original_img.GetDirection())
        
        # Save to temporary file
        output_path = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
        sitk.WriteImage(processed_img, output_path)
        
        return output_path, f"✓ Image saved to {output_path}"
    
    except Exception as e:
        return None, f"Error saving image: {str(e)}"


# Custom CSS for polished UI
custom_css = """
/* Root color variables */
:root {
    --bg-primary: #1a1d29;
    --bg-secondary: #252837;
    --bg-tertiary: #1e2130;
    --text-primary: #ffffff;
    --text-secondary: #e8e9f3;
    --text-tertiary: #b4b7c9;
    --text-muted: #9ca3bc;
    --text-dim: #6b7280;
    --accent: #6366f1;
    --accent-hover: #8b5cf6;
    --success: #10b981;
    --border-light: #3a3f52;
}

/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global styles */
* {
    font-family: 'Inter', sans-serif;
}

body, .gradio-container {
    background-color: var(--bg-primary) !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 40px !important;
}

/* Typography */
h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--text-primary) !important;
    margin-bottom: 8px;
}

h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-secondary) !important;
}

label, .label-wrap label {
    font-size: 16px !important;
    font-weight: 500 !important;
    color: var(--text-tertiary) !important;
    letter-spacing: 0;
}

.info-text {
    font-size: 14px;
    font-weight: 400;
    line-height: 1.5;
    color: var(--text-muted) !important;
}

/* Cards */
.block {
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    background: var(--bg-secondary) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    padding: 32px !important;
    transition: all 0.3s ease;
}

.block:hover {
    border-color: rgba(99, 102, 241, 0.3) !important;
    box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
}

/* Input/Output sections */
.input-container, .output-container {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    padding: 32px !important;
    margin-bottom: 40px !important;
}

/* File upload button */
.file-upload input {
    display: none;
}

.file-upload button, input[type="file"] + label, button.file-upload {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 12px 24px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: all 0.2s ease;
}

.file-upload button:hover, input[type="file"] + label:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3) !important;
}

.file-upload button:active, input[type="file"] + label:active {
    transform: scale(0.98);
}

/* Checkboxes */
input[type="checkbox"] {
    width: 20px !important;
    height: 20px !important;
    cursor: pointer;
    accent-color: var(--accent) !important;
    transition: all 0.2s ease;
}

input[type="checkbox"]:hover {
    transform: scale(1.1);
}

input[type="checkbox"]:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
}

.checkbox-wrap {
    display: flex;
    align-items: center;
    margin: 16px 0;
    gap: 8px;
}

.checkbox-wrap label {
    margin: 0 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    color: var(--text-tertiary) !important;
    cursor: pointer;
    transition: color 0.2s ease;
}

.checkbox-wrap input[type="checkbox"]:checked + label {
    color: var(--accent) !important;
}

/* Sliders */
input[type="range"] {
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(to right, var(--accent), var(--accent-hover));
    outline: none;
    -webkit-appearance: none;
    appearance: none;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
    transition: all 0.2s ease;
}

input[type="range"]::-webkit-slider-thumb:hover {
    width: 24px;
    height: 24px;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.6);
}

input[type="range"]::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: none;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
    transition: all 0.2s ease;
}

input[type="range"]::-moz-range-thumb:hover {
    width: 24px;
    height: 24px;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.6);
}

.range-slider-container {
    margin: 24px 0;
}

.slider-labels {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-muted) !important;
    margin-top: 8px;
}

.slider-input-box {
    width: 60px;
    padding: 8px;
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
    text-align: right;
    float: right;
    margin-top: -35px;
}

/* Buttons */
button.primary, button.gr-button-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 48px 32px !important;
    height: auto !important;
    width: 100% !important;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
}

button.primary:hover, button.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
}

button.primary:active, button.gr-button-primary:active {
    transform: translateY(0);
}

button.secondary, button.gr-button-secondary {
    background: var(--bg-secondary) !important;
    color: var(--text-tertiary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    cursor: pointer;
    transition: all 0.2s ease;
}

button.secondary:hover, button.gr-button-secondary:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* Text inputs and textareas */
input[type="text"], textarea, .textbox {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    padding: 12px !important;
    transition: all 0.2s ease;
}

input[type="text"]:focus, textarea:focus, .textbox:focus {
    outline: none;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

textarea, .textbox {
    background: var(--bg-tertiary) !important;
    color: var(--text-muted) !important;
    line-height: 1.5;
}

/* Image containers */
.output-image-container, .gr-image {
    background: #000000 !important;
    border-radius: 8px !important;
    border: 1px solid var(--border-light) !important;
    overflow: hidden;
    aspect-ratio: 16 / 9;
}

.image-label {
    font-size: 14px;
    color: var(--text-muted) !important;
    margin-bottom: 8px;
}

.slice-indicator {
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(0, 0, 0, 0.6);
    color: var(--text-primary);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
}

/* Processing log */
.log-container {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin-top: 40px;
}

.log-header {
    color: var(--accent) !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-light);
}

.log-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    margin: 8px 0;
    color: var(--text-muted) !important;
}

.log-entry.success {
    color: var(--success) !important;
}

.log-entry.success::before {
    content: "✓ ";
    color: var(--success) !important;
    font-weight: bold;
    margin-right: 4px;
}

/* Row and column spacing */
.row {
    gap: 40px !important;
    margin-bottom: 40px !important;
}

.column {
    gap: 32px !important;
}

/* Focus states for accessibility */
button:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
}

/* File size text */
.file-size-text {
    font-size: 14px;
    color: var(--text-muted) !important;
    margin-top: 16px;
}

/* Status message */
.status-message {
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 14px;
    margin-top: 16px;
}

.status-message.success {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success) !important;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-message.error {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444 !important;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 20px !important;
    }
    
    .row {
        flex-direction: column !important;
        gap: 24px !important;
    }
    
    h1 {
        font-size: 24px;
    }
    
    h2 {
        font-size: 18px;
    }
    
    .input-container, .output-container {
        padding: 20px !important;
    }
}
"""

with gr.Blocks(
    title="MRI Preprocessing Pipeline",
    css=custom_css,
    theme=gr.themes.Soft(),
) as demo:
    # Main title with emoji
    gr.HTML("""
    <div style="margin-bottom: 40px;">
        <h1 style="display: flex; align-items: center; margin-bottom: 8px;">
            <span style="margin-right: 8px; font-size: 36px;">🧠</span>
            MRI Data Preprocessing Pipeline
        </h1>
        <p class="info-text" style="margin: 0;">
            Advanced MRI preprocessing with bias field correction, head mask creation, and interactive visualization
        </p>
    </div>
    """)
    
    with gr.Row():
        # Input section (left column)
        with gr.Column(scale=1):
            gr.HTML('<div class="input-container"><h2>Input Section</h2>')
            
            uploaded_file = gr.File(
                label="Upload MRI Image (NIfTI)",
                file_types=[".nii", ".nii.gz"],
                file_count="single",
                type="filepath"
            )
            
            file_info = gr.Textbox(
                label="File Information",
                interactive=False,
                lines=2,
                value="No file uploaded yet"
            )
            
            gr.HTML('</div>')
        
        # Processing options (center-left column)
        with gr.Column(scale=1):
            gr.HTML('<div class="input-container"><h2>Processing Options</h2>')
            
            intensity_norm_checkbox = gr.Checkbox(
                value=True,
                label="Apply Intensity Normalization",
                show_label=True
            )
            
            mask_creation_checkbox = gr.Checkbox(
                value=True,
                label="Apply Head Mask Detection",
                show_label=True
            )
            
            bias_correction_checkbox = gr.Checkbox(
                value=True,
                label="Apply Bias Field Correction",
                show_label=True
            )
            
            gr.HTML('<div style="margin: 24px 0;"><label style="display: block; margin-bottom: 12px;">Slice to Display (%)</label>')
            
            with gr.Row():
                slice_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    show_label=False,
                    container=False
                )
                slice_value = gr.Textbox(
                    value="50",
                    lines=1,
                    max_lines=1,
                    interactive=True,
                    container=False
                )
            
            gr.HTML(
                '<div class="slider-labels"><span>0%</span><span>100%</span></div></div>'
            )
            
            process_btn = gr.Button(
                "🔄 Process Image",
                variant="primary",
                size="lg",
                scale=1
            )
            
            gr.HTML('</div>')
        
        # Output section (right column - 60% width)
        with gr.Column(scale=2):
            gr.HTML('<div class="output-container"><h2>Output Section</h2>')
            
            output_image = gr.Image(
                label="Before/After Comparison",
                type="pil",
                interactive=False
            )
            
            processing_log = gr.Textbox(
                label="Processing Log",
                lines=10,
                interactive=False,
                max_lines=15,
                elem_classes="log-container"
            )
            
            status_message = gr.Textbox(
                label="Status",
                interactive=False,
                lines=1,
                max_lines=1
            )
            
            with gr.Row():
                save_btn = gr.Button(
                    "💾 Save Processed Image",
                    variant="secondary",
                    scale=1
                )
                download_file = gr.File(
                    label="Download Processed Image",
                    interactive=False,
                    scale=1
                )
            
            gr.HTML('</div>')
    # Event handlers
    def update_file_info(file):
        """Update file information when file is uploaded"""
        if file is None:
            return "No file uploaded yet"
        
        try:
            file_path = file.name
            file_size = os.path.getsize(file_path)
            
            # Format file size
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            
            filename = Path(file_path).name
            return f"📄 {filename}\n{size_str}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def update_slice_from_slider(value):
        """Keep slider and textbox in sync"""
        return str(int(value))
    
    def update_slice_from_textbox(value):
        """Keep textbox and slider in sync"""
        try:
            val = int(value)
            val = max(0, min(100, val))  # Clamp between 0-100
            return val
        except:
            return 50
    
    def process_and_get_slice(file, intensity_norm, mask_create, bias_corr, slice_pct):
        """Process the MRI image with selected options"""
        if file is None:
            return None, "No processing: Please upload an image first", "❌ Error: No file uploaded"
        
        try:
            # Calculate actual slice from percentage
            raw_img = sitk.ReadImage(file, sitk.sitkFloat32)
            raw_img_arr = sitk.GetArrayFromImage(raw_img)
            max_slice = raw_img_arr.shape[0] - 1
            actual_slice = int((slice_pct / 100.0) * max_slice)
            
            img, log, arr, status = preprocess_mri(
                type('obj', (object,), {'name': file})(),
                bias_corr, mask_create, intensity_norm, actual_slice
            )
            
            # Store processed array for download
            demo.processed_array = arr
            demo.original_path = file
            
            return img, log, status
        except Exception as e:
            return None, f"Error: {str(e)}", f"❌ Error: {str(e)}"
    
    def save_and_download():
        """Save and download the processed image"""
        if not hasattr(demo, 'processed_array') or demo.processed_array is None:
            return None, "No processed image available"
        
        output_path, msg = save_processed_image(demo.processed_array, demo.original_path)
        return output_path, msg
    
    # Connect events
    uploaded_file.change(
        fn=update_file_info,
        inputs=[uploaded_file],
        outputs=[file_info]
    )
    
    slice_slider.change(
        fn=update_slice_from_slider,
        inputs=[slice_slider],
        outputs=[slice_value]
    )
    
    slice_value.change(
        fn=update_slice_from_textbox,
        inputs=[slice_value],
        outputs=[slice_slider]
    )
    
    process_btn.click(
        fn=process_and_get_slice,
        inputs=[uploaded_file, intensity_norm_checkbox, mask_creation_checkbox, bias_correction_checkbox, slice_slider],
        outputs=[output_image, processing_log, status_message]
    )
    
    save_btn.click(
        fn=save_and_download,
        outputs=[download_file, status_message]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
