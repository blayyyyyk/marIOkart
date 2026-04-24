import argparse
from pathlib import Path


def consolidate_python_files(directory_path, output_filename):
    # Convert string paths to pathlib.Path objects
    base_dir = Path(directory_path)
    output_file = Path(output_filename)

    # Ensure the target directory exists
    if not base_dir.is_dir():
        print(f"Error: The directory '{base_dir}' does not exist.")
        return

    count = 0

    # Open the output file in write mode
    with open(output_file, "w", encoding="utf-8") as out_f:
        # rglob('*.py') recursively searches for all python files
        for py_file in base_dir.rglob("*.py"):
            # Get the path relative to the base directory
            # (e.g., if base_dir is '.', this keeps 'src/folder/file.py')
            rel_path = py_file.relative_to(base_dir)

            # Skip the output file if it accidentally gets placed in the same directory
            # and ends with .py
            if py_file.resolve() == output_file.resolve():
                continue

            out_f.write(f"{rel_path}:\n")

            try:
                # Read the content of the current python file
                with open(py_file, encoding="utf-8") as in_f:
                    content = in_f.read()
                    out_f.write(content)

            except UnicodeDecodeError:
                out_f.write("# [Error: Could not decode file due to encoding issues]\n")
            except Exception as e:
                out_f.write(f"# [Error reading file: {e}]\n")

            # Add some spacing before the next file for readability
            out_f.write("\n\n")
            count += 1

    print(f"✅ Successfully consolidated {count} Python files into '{output_filename}'.")


if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Scrape Python files and copy them to a single text file.")
    parser.add_argument("directory", nargs="?", default=".", help="The root directory to scrape (defaults to current directory '.')")
    parser.add_argument("-o", "--output", default="consolidated_code.txt", help="The name of the output text file (defaults to 'consolidated_code.txt')")

    args = parser.parse_args()

    consolidate_python_files(args.directory, args.output)
