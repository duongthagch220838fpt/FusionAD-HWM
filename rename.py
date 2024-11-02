import os


folder_path = 'data/medicine/anomaly/low-light'


files = sorted(os.listdir(folder_path))

for index, filename in enumerate(files, start=0):
    # Get the file extension
    file_extension = os.path.splitext(filename)[1]


    new_name = f"{index:03}{file_extension}"


    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_name)


    os.rename(old_file, new_file)
    print(f"Renamed '{filename}' to '{new_name}'")