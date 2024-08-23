import os

def modify_text_files(directory):
    # Walk through the directory and all its subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the file is a text file
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                
                # Read the contents of the file
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                # Check if the file has at least two lines
                if len(lines) >= 2:
                    # Modify the second line if it matches the specified text
                    if lines[1].strip() == "Classical Embeddings: GloVe 50-d":
                        lines[1] = "Classical Embeddings: GloVe 300-d Common Crawl\n"
                        
                        # Write the modified contents back to the file
                        with open(file_path, 'w') as file:
                            file.writelines(lines)
                            print(f"Modified: {file_path}")


# Example usage
directory_path = 'C:\\Users\\ucl_l\\Documents\\GitHub\\QFsl\\runs'
modify_text_files(directory_path)
