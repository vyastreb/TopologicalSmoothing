#!/usr/bin/env python3
"""
Test cases for topology-preserving smoothing algorithm.

Creates various test shapes and compares results across different parameters:
- Connectivity: 4 and 8
- Algorithm: ASFT and ASFT-MED (with medial axis)
- Scaling: 1 and 4
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from topology_smoothing import (
    asft, asftmed, binarize_image, scale_binary_image
)

# Output directory
OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image parameters
SHAPE_SIZE = 64  # Size of test images
BASE_RADIUS = 3  # Base smoothing radius (for scale=1)


def create_rectangle(width=40, height=20):
    """Create a grid-aligned rectangle."""
    img = np.zeros((SHAPE_SIZE, SHAPE_SIZE), dtype=np.uint8)
    y0 = (SHAPE_SIZE - height) // 2
    x0 = (SHAPE_SIZE - width) // 2
    img[y0:y0+height, x0:x0+width] = 255
    return img


def create_square(size=30):
    """Create a grid-aligned square."""
    img = np.zeros((SHAPE_SIZE, SHAPE_SIZE), dtype=np.uint8)
    y0 = (SHAPE_SIZE - size) // 2
    x0 = (SHAPE_SIZE - size) // 2
    img[y0:y0+size, x0:x0+size] = 255
    return img


def create_l_shape(width=30, height=30, thickness=10):
    """Create a grid-aligned L-shape."""
    img = np.zeros((SHAPE_SIZE, SHAPE_SIZE), dtype=np.uint8)
    y0 = (SHAPE_SIZE - height) // 2
    x0 = (SHAPE_SIZE - width) // 2
    # Vertical bar
    img[y0:y0+height, x0:x0+thickness] = 255
    # Horizontal bar
    img[y0+height-thickness:y0+height, x0:x0+width] = 255
    return img


def create_two_squares_edge_touch(size=15):
    """Create two squares touching at a FULL edge (many 4-neighbor touches).
    
    The squares share a common edge, so they form ONE connected component
    for both 4-connectivity and 8-connectivity.
    """
    img = np.zeros((SHAPE_SIZE, SHAPE_SIZE), dtype=np.uint8)
    # First square (left)
    y0, x0 = 20, 10
    img[y0:y0+size, x0:x0+size] = 255
    # Second square (right, sharing right edge of first = left edge of second)
    y1, x1 = y0, x0 + size  # Same y, x starts where first ends
    img[y1:y1+size, x1:x1+size] = 255
    return img


def create_two_squares_single_point_touch(size=15):
    """Create two squares touching at exactly ONE 4-neighbor pixel.
    
    The squares touch at a single point that is a 4-neighbor (horizontal/vertical),
    not a diagonal. They form ONE connected component for both connectivities.
    
    Layout:
        Square 1 (top-left)
                 [corner pixel]
        Square 2 (bottom-right, shifted so only corners touch at 4-neighbor)
    """
    img = np.zeros((SHAPE_SIZE, SHAPE_SIZE), dtype=np.uint8)
    # First square (top-left area)
    y0, x0 = 10, 10
    img[y0:y0+size, x0:x0+size] = 255
    # Second square - positioned so ONLY the bottom-right corner of first
    # touches the top-left corner of second via a single 4-neighbor
    # First square ends at (y0+size-1, x0+size-1) = (24, 24)
    # Second square starts so its top-left is at (25, 24) - shares x, adjacent y
    y1, x1 = y0 + size, x0 + size - 1  # Single pixel touch on x=24
    img[y1:y1+size, x1:x1+size] = 255
    return img


def create_two_squares_diagonal(size=15):
    """Create two squares touching ONLY at diagonal (8-neighbor touch only).
    
    The squares touch only at one diagonal point, so they form:
    - TWO components with 4-connectivity (diagonal not counted as neighbor)
    - ONE component with 8-connectivity (diagonal IS a neighbor)
    """
    img = np.zeros((SHAPE_SIZE, SHAPE_SIZE), dtype=np.uint8)
    # First square (top-left)
    y0, x0 = 15, 15
    img[y0:y0+size, x0:x0+size] = 255
    # Second square (bottom-right, diagonal touch only)
    # Starts exactly where first ends on both axes
    y1, x1 = y0 + size, x0 + size
    img[y1:y1+size, x1:x1+size] = 255
    return img


def create_rotated_square(size=25, angle_deg=45):
    """Create a square rotated by given angle."""
    # Create larger canvas for rotation
    canvas_size = SHAPE_SIZE * 2
    img = Image.new('L', (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)
    
    # Calculate square corners
    cx, cy = canvas_size // 2, canvas_size // 2
    half = size // 2
    
    # Square corners before rotation
    corners = [
        (-half, -half),
        (half, -half),
        (half, half),
        (-half, half)
    ]
    
    # Rotate corners
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotated = []
    for x, y in corners:
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        rotated.append((rx, ry))
    
    # Draw filled polygon
    draw.polygon(rotated, fill=255)
    
    # Crop to final size
    img = img.crop((
        (canvas_size - SHAPE_SIZE) // 2,
        (canvas_size - SHAPE_SIZE) // 2,
        (canvas_size + SHAPE_SIZE) // 2,
        (canvas_size + SHAPE_SIZE) // 2
    ))
    
    return np.array(img)


def process_shape(img, connex, use_medial, scale):
    """Process a shape with given parameters.
    
    The radius is scaled proportionally with the scale factor:
    - scale=1: radius = BASE_RADIUS (e.g., 3)
    - scale=4: radius = BASE_RADIUS * 4 (e.g., 12)
    """
    # Scale the image
    if scale > 1:
        scaled = scale_binary_image(img, scale)
    else:
        scaled = img.copy()
    
    # Scale the radius proportionally
    radius = BASE_RADIUS * scale
    
    # Apply algorithm
    if use_medial:
        result = asftmed(scaled, connex, radius)
    else:
        result = asft(scaled, connex, radius)
    
    return result


def create_comparison_panel(shape_name, original_img):
    """Create a comparison panel for a shape with all parameter combinations."""
    
    # Parameters to test
    scales = [1, 4]
    connectivities = [4, 8]
    algorithms = [('ASFT', False), ('ASFT-MED', True)]
    
    # Calculate panel dimensions
    n_cols = 1 + len(scales) * len(connectivities) * len(algorithms)  # Original + combinations
    n_rows = 1
    
    # Cell size (larger for scaled images)
    cell_size = max(SHAPE_SIZE * max(scales), 256)
    padding = 10
    header_height = 60
    
    # Actual layout: Original on top-left, then grid of results
    # Row 1: Original
    # Row 2+: Results organized as: Scale 1 (C4-ASFT, C4-MED, C8-ASFT, C8-MED), Scale 4 (...)
    
    results_per_row = 4  # C4-ASFT, C4-MED, C8-ASFT, C8-MED
    n_result_rows = len(scales)
    
    total_width = padding + cell_size + padding + results_per_row * (cell_size + padding)
    total_height = header_height + padding + (1 + n_result_rows) * (cell_size + padding + 30)
    
    # Create panel
    panel = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_title = font
    
    # Title
    draw.text((padding, 10), f"Test: {shape_name}", fill=(0, 0, 0), font=font_title)
    draw.text((padding, 35), f"Base radius: {BASE_RADIUS} (scaled with image)", fill=(100, 100, 100), font=font_small)
    
    y_offset = header_height + padding
    
    # Draw original image
    orig_resized = Image.fromarray(original_img).resize((cell_size, cell_size), Image.NEAREST)
    panel.paste(orig_resized, (padding, y_offset + 20))
    draw.text((padding, y_offset), "Original", fill=(0, 0, 0), font=font)
    draw.text((padding, y_offset + cell_size + 22), f"{original_img.shape[1]}x{original_img.shape[0]}", 
              fill=(100, 100, 100), font=font_small)
    
    # Process and draw results
    x_base = padding + cell_size + padding * 2
    
    for scale_idx, scale in enumerate(scales):
        y_pos = y_offset + scale_idx * (cell_size + padding + 40)
        
        # Scale label
        draw.text((x_base, y_pos - 18), f"Scale {scale}x", fill=(0, 0, 150), font=font)
        
        col = 0
        for connex in connectivities:
            for algo_name, use_medial in algorithms:
                x_pos = x_base + col * (cell_size + padding)
                
                # Process
                print(f"  Processing {shape_name}: scale={scale}, connex={connex}, {algo_name}...")
                result = process_shape(original_img, connex, use_medial, scale)
                
                # Resize result to cell_size for display
                result_img = Image.fromarray(result)
                if result.shape[0] != cell_size:
                    result_img = result_img.resize((cell_size, cell_size), Image.NEAREST)
                
                panel.paste(result_img, (x_pos, y_pos + 20))
                
                # Label
                label = f"C{connex} {algo_name}"
                draw.text((x_pos, y_pos), label, fill=(0, 0, 0), font=font_small)
                draw.text((x_pos, y_pos + cell_size + 22), 
                          f"{result.shape[1]}x{result.shape[0]}", 
                          fill=(100, 100, 100), font=font_small)
                
                col += 1
    
    return panel


def create_compact_panel(shape_name, original_img):
    """Create a more compact comparison panel."""
    
    # Parameters to test
    scales = [1, 4]
    connectivities = [4, 8]
    algorithms = [('ASFT', False), ('MED', True)]
    
    # Cell dimensions
    base_cell = 80
    padding = 8
    label_width = 45  # Width for row labels (s=X, r=Y)
    header_height = 20
    
    n_result_cols = len(connectivities) * len(algorithms)
    n_rows = len(scales)
    
    # Calculate sizes
    orig_cell = base_cell
    result_cell = base_cell
    
    # Layout: Title | Original | Label column | Results grid
    total_width = padding + orig_cell + padding + label_width + n_result_cols * (result_cell + padding)
    total_height = padding + 20 + header_height + n_rows * (result_cell + padding) + padding
    
    # Create panel
    panel = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    
    # Font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_title = font
    
    # Title
    draw.text((padding, padding), shape_name, fill=(0, 0, 0), font=font_title)
    
    y_start = padding + 22
    
    # Draw original image (vertically centered across result rows)
    orig_resized = Image.fromarray(original_img).resize((orig_cell, orig_cell), Image.NEAREST)
    orig_y = y_start + header_height + (n_rows * (result_cell + padding) - orig_cell) // 2
    panel.paste(orig_resized, (padding, orig_y))
    draw.text((padding + orig_cell // 2 - 20, orig_y + orig_cell + 3), "Original", fill=(0, 0, 0), font=font)
    
    # X position for labels and results
    label_x = padding + orig_cell + padding
    results_x = label_x + label_width
    
    # Column headers
    headers = ["C4-ASFT", "C4-MED", "C8-ASFT", "C8-MED"]
    for i, header in enumerate(headers):
        x_pos = results_x + i * (result_cell + padding)
        draw.text((x_pos + 5, y_start), header, fill=(80, 80, 80), font=font)
    
    # Process and draw results
    for scale_idx, scale in enumerate(scales):
        y_pos = y_start + header_height + scale_idx * (result_cell + padding)
        
        # Row label (scale and radius) - in dedicated label column
        radius = BASE_RADIUS * scale
        draw.text((label_x + 5, y_pos + result_cell // 2 - 12), f"s={scale}", fill=(0, 0, 150), font=font)
        draw.text((label_x + 5, y_pos + result_cell // 2 + 2), f"r={radius}", fill=(0, 100, 0), font=font)
        
        col = 0
        for connex in connectivities:
            for algo_name, use_medial in algorithms:
                x_pos = results_x + col * (result_cell + padding)
                
                # Process
                result = process_shape(original_img, connex, use_medial, scale)
                
                # Resize for display
                result_img = Image.fromarray(result).resize((result_cell, result_cell), Image.NEAREST)
                panel.paste(result_img, (x_pos, y_pos))
                
                col += 1
    
    return panel


def main():
    print("Creating test shapes...")
    
    # Define test shapes
    shapes = [
        ("1_Rectangle", create_rectangle(40, 20)),
        ("2_Square", create_square(30)),
        ("3_L_Shape", create_l_shape(30, 30, 10)),
        ("4_Squares_Edge_Touch", create_two_squares_edge_touch(15)),
        ("5_Squares_Single_Point_Touch", create_two_squares_single_point_touch(15)),
        ("6_Squares_Diagonal_Touch", create_two_squares_diagonal(15)),
        ("7_Square_Rotated_30deg", create_rotated_square(25, 30)),
        ("8_Square_Rotated_45deg", create_rotated_square(25, 45)),
    ]
    
    # Create individual panels
    all_panels = []
    for name, img in shapes:
        print(f"\nProcessing {name}...")
        
        # Save original
        Image.fromarray(img).save(os.path.join(OUTPUT_DIR, f"{name}_original.png"))
        
        # Create comparison panel
        panel = create_compact_panel(name, img)
        panel.save(os.path.join(OUTPUT_DIR, f"{name}_comparison.png"))
        all_panels.append((name, panel))
        print(f"  Saved {name}_comparison.png")
    
    # Create combined image with all panels
    print("\nCreating combined overview...")
    
    # Stack panels vertically
    total_height = sum(p.size[1] for _, p in all_panels) + 10 * (len(all_panels) - 1)
    max_width = max(p.size[0] for _, p in all_panels)
    
    combined = Image.new('RGB', (max_width, total_height), (240, 240, 240))
    y = 0
    for name, panel in all_panels:
        combined.paste(panel, (0, y))
        y += panel.size[1] + 10
    
    combined.save(os.path.join(OUTPUT_DIR, "all_tests_overview.png"))
    print(f"Saved all_tests_overview.png")
    
    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

