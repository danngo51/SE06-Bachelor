import os
import zipfile

def zip_directory(folder_path, output_path=None):
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"'{folder_path}' is not a valid directory.")

    # Default zip filename
    if output_path is None:
        output_path = folder_path.rstrip("/\\") + ".zip"

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)
    
    print(f"Folder zipped successfully to: {output_path}")

# Example usage
if __name__ == "__main__":
    folder_to_zip = './Informer2020'  # change this to the folder you want to zip
    zip_directory(folder_to_zip)
