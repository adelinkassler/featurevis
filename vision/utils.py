import os

def ensure_dir_exists_for_file(file_path):
    """
    Ensures the directory exists for a given file path.

    Args:
        file_path (str): The path to the file.

    Raises:
        OSError: If there is an error creating the directory.
    """

    # Given a file path, make the directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        try:
            # Create the directory recursively
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Error creating directory: {e}")
