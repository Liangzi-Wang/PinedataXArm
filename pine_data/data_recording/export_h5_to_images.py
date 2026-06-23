#!/usr/bin/env python3
"""
Script to export camera data from HDF5 file to PNG images.

Usage:
    python export_h5_to_images.py <h5_file_path> [output_folder]
    
Example:
    python export_h5_to_images.py /home/pine/pine_data/recordings/20251215172808/camera_data.h5
    python export_h5_to_images.py camera_data.h5 ./exported_images
"""

import h5py
import numpy as np
import os
import sys
from PIL import Image
import cv2


def export_h5_to_images(h5_file_path, output_folder=None):
    """
    Export RGB and depth images from HDF5 file to PNG format.
    
    Args:
        h5_file_path: Path to the HDF5 file
        output_folder: Output folder path (default: same folder as h5 file with _images suffix)
    """
    
    # Check if file exists
    if not os.path.exists(h5_file_path):
        print(f"Error: File not found: {h5_file_path}")
        return
    
    # Determine output folder
    if output_folder is None:
        h5_dir = os.path.dirname(h5_file_path)
        h5_name = os.path.splitext(os.path.basename(h5_file_path))[0]
        output_folder = os.path.join(h5_dir, f"{h5_name}_images")
    
    # Create output directories
    rgb_folder = os.path.join(output_folder, "rgb")
    depth_folder = os.path.join(output_folder, "depth")
    depth_viz_folder = os.path.join(output_folder, "depth_visualization")
    
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(depth_viz_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Exporting images from HDF5 file")
    print(f"{'='*60}")
    print(f"Input file: {h5_file_path}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}\n")
    
    # Load data from HDF5
    with h5py.File(h5_file_path, 'r') as f:
        print("Loading data from HDF5...")
        
        # Load RGB images
        rgb_data = f['rgb_hand'][:]
        print(f"RGB data shape: {rgb_data.shape}")
        
        # Load depth images
        depth_data = f['depth_hand'][:]
        print(f"Depth data shape: {depth_data.shape}")
        
        # Load metadata
        print("\nMetadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        num_frames = len(rgb_data)
        print(f"\nTotal frames to export: {num_frames}")
        print()
    
    # Export RGB images
    print("Exporting RGB images...")
    for i in range(num_frames):
        rgb_image = rgb_data[i]
        
        # Save as PNG
        rgb_path = os.path.join(rgb_folder, f"frame_{i:06d}.png")
        Image.fromarray(rgb_image.astype(np.uint8)).save(rgb_path)
        
        if (i + 1) % 50 == 0 or i == num_frames - 1:
            print(f"  Exported {i + 1}/{num_frames} RGB images", end='\r')
    print(f"\n✓ RGB images saved to: {rgb_folder}")
    
    # Export depth images
    print("\nExporting depth images...")
    for i in range(num_frames):
        depth_image = depth_data[i]
        
        # Save raw depth as 16-bit PNG
        depth_path = os.path.join(depth_folder, f"frame_{i:06d}.png")
        cv2.imwrite(depth_path, depth_image.astype(np.uint16))
        
        # Create colorized visualization for easier viewing
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_colorized = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        depth_viz_path = os.path.join(depth_viz_folder, f"frame_{i:06d}.png")
        cv2.imwrite(depth_viz_path, depth_colorized)
        
        if (i + 1) % 50 == 0 or i == num_frames - 1:
            print(f"  Exported {i + 1}/{num_frames} depth images", end='\r')
    print(f"\n✓ Depth images (raw) saved to: {depth_folder}")
    print(f"✓ Depth visualization saved to: {depth_viz_folder}")
    
    # Create a summary file
    summary_path = os.path.join(output_folder, "export_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Export Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Source file: {h5_file_path}\n")
        f.write(f"Total frames: {num_frames}\n")
        f.write(f"RGB image shape: {rgb_data[0].shape}\n")
        f.write(f"Depth image shape: {depth_data[0].shape}\n")
        f.write(f"\nOutput folders:\n")
        f.write(f"  RGB: {rgb_folder}\n")
        f.write(f"  Depth (raw): {depth_folder}\n")
        f.write(f"  Depth (visualization): {depth_viz_folder}\n")
    
    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"Total frames exported: {num_frames}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_h5_to_images.py <h5_file_path> [output_folder]")
        print("\nExample:")
        print("  python export_h5_to_images.py /home/pine/pine_data/recordings/20251215172808/camera_data.h5")
        print("  python export_h5_to_images.py camera_data.h5 ./exported_images")
        sys.exit(1)
    
    h5_file_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    export_h5_to_images(h5_file_path, output_folder)


if __name__ == "__main__":
    main()
