import os
from pathlib import Path

def copy_folder_structure(source_root, dest_root, max_depth=4):
    """
    Copies a directory structure from a source to a destination, ignoring files.

    Args:
        source_root (Path): The root directory to copy from.
        dest_root (Path): The root directory to copy to.
        max_depth (int): The maximum number of levels to descend.
    """
    print(f"Starting folder copy from '{source_root}' to '{dest_root}' (max depth: {max_depth})...")

    # Ensure the destination root exists
    dest_root.mkdir(exist_ok=True)

    for root, dirs, files in os.walk(source_root):
        # Calculate the current depth
        current_depth = len(Path(root).relative_to(source_root).parts)

        # Stop if we have reached the maximum depth
        if current_depth >= max_depth:
            # By clearing the dirs list, os.walk will not descend further
            dirs[:] = []
            continue

        # For each directory found, create a corresponding one in the destination
        for dir_name in dirs:
            source_path = Path(root) / dir_name
            # Get the relative path from the source root to the current directory
            relative_path = source_path.relative_to(source_root)
            # Create the full destination path
            dest_path = dest_root / relative_path

            print(f"Creating directory: {dest_path}")
            # Handle potential long paths on Windows by using a special prefix.
            if os.name == 'nt':
                # The `\\?\` prefix allows paths to exceed the 260-character limit.
                # `os.makedirs` supports this prefix, while `pathlib.mkdir` may not.
                try:
                    os.makedirs("\\\\?\\" + str(dest_path.resolve()), exist_ok=True)
                except OSError as e:
                    print(f"Error creating directory {dest_path}: {e}")
            else:
                # On non-Windows systems, the standard method is fine.
                dest_path.mkdir(parents=True, exist_ok=True)

    print("\nFolder structure copy complete.")
    print(f"Check the '{dest_root}' directory.")

if __name__ == '__main__':
    # Define the source and destination paths
    # Note: os.path.expanduser('~') correctly finds the user's home directory (e.g., C:\Users\Alan)
    source_directory = Path(os.path.expanduser('~')) / "OneDrive" / "Documents"
    destination_directory = Path(__file__).parent / "final_documents"

    # Check if the source directory exists before running
    if source_directory.is_dir():
        copy_folder_structure(source_directory, destination_directory, max_depth=4)
    else:
        print(f"Error: Source directory '{source_directory}' not found.")
        print("Please check that the path is correct.")
