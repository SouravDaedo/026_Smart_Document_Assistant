"""
Interactive Dataset Explorer for DocLayNet and other layout detection datasets.
Allows browsing samples by index instead of random selection.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter
import argparse

class DatasetExplorer:
    """Interactive explorer for layout detection datasets."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.doclaynet_data = None
        self.available_images = []
        self.load_data()
    
    def load_data(self):
        """Load DocLayNet annotations and find available images."""
        doclaynet_dir = self.data_dir / "doclaynet"
        train_annotations = doclaynet_dir / "COCO" / "train.json"
        
        if train_annotations.exists():
            with open(train_annotations, 'r') as f:
                self.doclaynet_data = json.load(f)
            
            # Find available images
            image_dir = doclaynet_dir / "PNG"
            if image_dir.exists():
                self.available_images = [
                    img for img in self.doclaynet_data['images']
                    if (image_dir / img['file_name']).exists()
                ]
            
            print(f"✅ Loaded {len(self.doclaynet_data['images'])} total images")
            print(f"📁 Found {len(self.available_images)} available images")
        else:
            print(" DocLayNet data not found. Run: python scripts/download_training_data.py --dataset doclaynet")
    
    def show_sample(self, index=0):
        """Show sample by index (0-based)."""
        if not self.available_images:
            print("❌ No images available")
            return
        
        if index >= len(self.available_images):
            print(f"❌ Index {index} out of range. Available: 0-{len(self.available_images)-1}")
            return
        
        # Get image by index
        image_info = self.available_images[index]
        image_id = image_info['id']
        
        # Get annotations for this image
        annotations = [ann for ann in self.doclaynet_data['annotations'] if ann['image_id'] == image_id]
        
        print(f"🖼️  Sample #{index+1}: {image_info['file_name']}")
        print(f"📊 Annotations: {len(annotations)}")
        
        # Show categories
        category_names = {cat['id']: cat['name'] for cat in self.doclaynet_data['categories']}
        image_categories = [category_names[ann['category_id']] for ann in annotations]
        print(f"📋 Categories: {list(set(image_categories))}")
        
        # Visualize
        self.visualize_image(image_info, annotations)
    
    def visualize_image(self, image_info, annotations):
        """Visualize image with bounding boxes."""
        image_dir = self.data_dir / "doclaynet" / "PNG"
        image_path = image_dir / image_info['file_name']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Category colors
        categories = self.doclaynet_data['categories']
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        color_map = {cat['id']: tuple(int(c*255) for c in colors[i][:3]) for i, cat in enumerate(categories)}
        
        # Draw bounding boxes
        for ann in annotations:
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']
            color = color_map.get(category_id, (255, 0, 0))
            
            # Draw rectangle
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
            
            # Draw category label
            category_name = next((cat['name'] for cat in categories if cat['id'] == category_id), 'Unknown')
            draw.text((x, max(0, y-20)), category_name, fill=color)
        
        # Display
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Sample: {image_info['file_name']} ({image.size[0]}x{image.size[1]})")
        plt.tight_layout()
        plt.show()
    
    def show_category_stats(self):
        """Show category distribution statistics."""
        if not self.doclaynet_data:
            return
        
        category_counts = Counter([ann['category_id'] for ann in self.doclaynet_data['annotations']])
        category_names = {cat['id']: cat['name'] for cat in self.doclaynet_data['categories']}
        
        # Create DataFrame
        stats_data = []
        for cat_id, count in category_counts.items():
            stats_data.append({
                'Category': category_names[cat_id],
                'Count': count,
                'Percentage': f"{count/len(self.doclaynet_data['annotations'])*100:.1f}%"
            })
        
        df = pd.DataFrame(stats_data).sort_values('Count', ascending=False)
        print("📊 Category Distribution:")
        print(df.to_string(index=False))
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(df['Category'], df['Count'])
        plt.title('DocLayNet Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Number of Annotations')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def browse_samples(self, start=0, count=5):
        """Browse multiple samples starting from index."""
        print(f"🔍 Browsing samples {start+1} to {start+count}:")
        
        for i in range(start, min(start + count, len(self.available_images))):
            image_info = self.available_images[i]
            image_id = image_info['id']
            annotations = [ann for ann in self.doclaynet_data['annotations'] if ann['image_id'] == image_id]
            
            category_names = {cat['id']: cat['name'] for cat in self.doclaynet_data['categories']}
            image_categories = list(set([category_names[ann['category_id']] for ann in annotations]))
            
            print(f"  {i+1:3d}. {image_info['file_name'][:50]:<50} | {len(annotations):2d} annotations | {', '.join(image_categories[:3])}")
    
    def find_samples_with_category(self, category_name, limit=10):
        """Find samples containing specific category."""
        if not self.doclaynet_data:
            return
        
        # Find category ID
        category_id = None
        for cat in self.doclaynet_data['categories']:
            if cat['name'].lower() == category_name.lower():
                category_id = cat['id']
                break
        
        if category_id is None:
            print(f"❌ Category '{category_name}' not found")
            return
        
        # Find images with this category
        matching_images = []
        for img in self.available_images:
            annotations = [ann for ann in self.doclaynet_data['annotations'] 
                         if ann['image_id'] == img['id'] and ann['category_id'] == category_id]
            if annotations:
                matching_images.append((img, len(annotations)))
        
        print(f"🔍 Found {len(matching_images)} images with '{category_name}' category:")
        
        for i, (img, count) in enumerate(matching_images[:limit]):
            img_index = self.available_images.index(img)
            print(f"  {img_index+1:3d}. {img['file_name'][:50]:<50} | {count} {category_name} annotations")


def main():
    parser = argparse.ArgumentParser(description="Explore layout detection datasets")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--sample", type=int, help="Show specific sample by index (1-based)")
    parser.add_argument("--browse", nargs=2, type=int, metavar=('START', 'COUNT'), 
                       help="Browse samples from START for COUNT items")
    parser.add_argument("--category", help="Find samples with specific category")
    parser.add_argument("--stats", action="store_true", help="Show category statistics")
    
    args = parser.parse_args()
    
    explorer = DatasetExplorer(args.data_dir)
    
    if args.sample:
        explorer.show_sample(args.sample - 1)  # Convert to 0-based
    elif args.browse:
        start, count = args.browse
        explorer.browse_samples(start - 1, count)  # Convert to 0-based
    elif args.category:
        explorer.find_samples_with_category(args.category)
    elif args.stats:
        explorer.show_category_stats()
    else:
        # Interactive mode
        print("🔍 Dataset Explorer - Interactive Mode")
        print("Available commands:")
        print("  sample <n>     - Show sample n (1-based)")
        print("  browse <start> <count> - Browse samples")
        print("  category <name> - Find samples with category")
        print("  stats          - Show statistics")
        print("  quit           - Exit")
        
        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue
                
                if cmd[0] == 'quit':
                    break
                elif cmd[0] == 'sample' and len(cmd) == 2:
                    explorer.show_sample(int(cmd[1]) - 1)
                elif cmd[0] == 'browse' and len(cmd) == 3:
                    explorer.browse_samples(int(cmd[1]) - 1, int(cmd[2]))
                elif cmd[0] == 'category' and len(cmd) == 2:
                    explorer.find_samples_with_category(cmd[1])
                elif cmd[0] == 'stats':
                    explorer.show_category_stats()
                else:
                    print("❌ Invalid command")
            except (ValueError, IndexError):
                print("❌ Invalid command format")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
