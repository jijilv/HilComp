import numpy as np
import os

def categorize_component(key):
    if 'xyz' in key:
        return 'pos'
    elif 'features' in key:
        return 'color'
    elif 'scaling' in key or 'rotation' in key:
        return 'shape'
    elif 'opa' in key:
        return 'opa'
    elif 'indices' in key:
        return 'indices'
    else:
        return 'others'

def calculate_npz_memory_ratios_with_categories(file_path):

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return {}

    try:
        npz_data = np.load(file_path)
        
        print("Components in the .npz file:", npz_data.files)
        
        category_sizes = {}
        for key in npz_data.files:
            category = categorize_component(key)
            size = npz_data[key].nbytes
            if category in category_sizes:
                category_sizes[category] += size
            else:
                category_sizes[category] = size
        
        total_size = sum(category_sizes.values())
        print(f"Total memory size: {total_size} bytes")
        
        if total_size == 0:
            print("Error: The total size of the .npz file components is zero.")
            return {}
        
        category_ratios = {category: size / total_size for category, size in category_sizes.items()}
        
        print("Memory usage ratios by categories:")
        for category, ratio in category_ratios.items():
            print(f"  {category}: {ratio * 100:.2f}%")
        
        return category_ratios

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return {}

if __name__ == "__main__":
    file_path = "/mnt/hy/output/Transamerica-hi/point_cloud/iteration_80000/point_cloud.npz"
    memory_ratios = calculate_npz_memory_ratios_with_categories(file_path)
    print("\nFinal categorized memory ratios (as dictionary):")
    print(memory_ratios)
