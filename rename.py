import os


# Define the folder name patterns and their new names
rename_map = {
    'low': 'low-light',
    'well': 'well-light',
    'Low': 'low-light',
    'Well': 'well-light',
}
def rename_folders(directory_path, rename_map=rename_map):
    # List all folders in the directory
    existing_folders = os.listdir(directory_path)

    for prefix, new_name in rename_map.items():
        # Find all folders that start with the prefix
        matching_folders = [f for f in existing_folders if f.startswith(prefix)]
        
        # Check for multiple matches
        if len(matching_folders) > 1:
            print(f"Multiple folders found starting with '{prefix}': {matching_folders}. Skipping rename.")
        elif len(matching_folders) == 1:
            old_folder = matching_folders[0]
            new_folder_path = os.path.join(directory_path, new_name)
            
            # Check if the new folder name already exists
            if new_name not in existing_folders:
                old_folder_path = os.path.join(directory_path, old_folder)
                os.rename(old_folder_path, new_folder_path)
                print(f"Renamed '{old_folder}' to '{new_name}'")
            else:
                print(f"Folder '{new_name}' already exists. Skipping rename for '{prefix}'.")
        else:
            print(f"No folder starting with '{prefix}' found.")


def check_file_alignment(directory_path, gt_path=None):
    # Define paths to the well-light and low-light folders
    well_light_path = os.path.join(directory_path, 'well-light')
    low_light_path = os.path.join(directory_path, 'low-light')
    
    # Check if well-light and low-light directories exist
    if not all(os.path.exists(path) for path in [well_light_path, low_light_path]):
        print("One or both required directories ('well-light' or 'low-light') do not exist.")
        return
    
    # List base filenames (without extensions) in well-light and low-light directories
    well_light_files = {os.path.splitext(f)[0] for f in os.listdir(well_light_path)}
    low_light_files = {os.path.splitext(f)[0] for f in os.listdir(low_light_path)}
    
    # If gt_path is provided, include it in the comparison

    if gt_path:
        # Check if the gt directory exists
        if not os.path.exists(gt_path):
            print(f"The 'gt' directory '{gt_path}' does not exist.")
            return
        
        # List base filenames in gt directory
        gt_files = {os.path.splitext(f)[0] for f in os.listdir(gt_path)}
        
        # Check for alignment across all three directories
        if well_light_files == low_light_files == gt_files:
            print("All files are aligned across 'Well-light', 'Low-light', and 'gt' directories (ignoring extensions).")
        else:
            # Find mismatched files
            missing_in_well = (low_light_files | gt_files) - well_light_files
            missing_in_low = (well_light_files | gt_files) - low_light_files
            missing_in_gt = (well_light_files | low_light_files) - gt_files
            
            # Report misalignments
            if missing_in_well:
                print("Files present in 'Low-light' or 'gt' but missing in 'Well-light':")
                for file in missing_in_well:
                    print(f" - {file}")
            
            if missing_in_low:
                print("Files present in 'Well-light' or 'gt' but missing in 'Low-light':")
                for file in missing_in_low:
                    print(f" - {file}")
            
            if missing_in_gt:
                print("Files present in 'Well-light' or 'Low-light' but missing in 'gt':")
                for file in missing_in_gt:
                    print(f" - {file}")
    else:
        # Check for alignment between just well-light and low-light
        if well_light_files == low_light_files:
            print("All files are aligned between 'Well-light' and 'Low-light' directories.")
        else:
            # Find mismatched files between well-light and low-light
            missing_in_well = low_light_files - well_light_files
            missing_in_low = well_light_files - low_light_files
            
            # Report misalignments
            if missing_in_well:
                print("Files present in 'Low-light' but missing in 'Well-light':")
                for file in missing_in_well:
                    print(f" - {file}")
            
            if missing_in_low:
                print("Files present in 'Well-light' but missing in 'Low-light':")
                for file in missing_in_low:
                    print(f" - {file}")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check file alignment between well-light and low-light folders')
    parser.add_argument('directory_path', type=str, help='Path to the directory containing well-light and low-light folders', default='data')
    parser.add_argument('--gt_path', type=str, help='Path to the directory containing ground truth images', default='data/medicine_pack')
    args = parser.parse_args()
    # Folder structure is as follows:
    # data
    # ├── annotation
    # │   ├── *.json
    # ├── anomaly
    # │   ├── low_light
    # │   │   ├── *.jpg
    # │   ├── well_light
    # │   │   ├── *.jpg
    # ├── normal
    # │   ├── wow_light
    # │   │   ├── *.jpg
    # │   ├── well_light
    # │   │   ├── *.jpg
    # └── gt
    #     ├── *.png


    rename_folders(os.path.join(args.directory_path, 'anomaly'), rename_map)
    rename_folders(os.path.join(args.directory_path, 'normal'), rename_map)

    if args.gt_path:
        check_file_alignment(os.path.join(args.directory_path, 'anomaly'), os.path.join(args.gt_path, 'gt'))
        check_file_alignment(os.path.join(args.directory_path,'normal'))