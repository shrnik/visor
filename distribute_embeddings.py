
from create_embeddings import process_folder

def getFolderList(baseDir):
    import os

    folderList = []
    for entry in os.listdir(baseDir):
        fullPath = os.path.join(baseDir, entry)
        if os.path.isdir(fullPath):
            folderList.append(fullPath)
    return folderList

def process_folders(folders):
    for folder in folders:
        process_folder(folder)
if __name__ == "__main__":
    import argparse
    import os

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--node", type=int, required=True, help="Node number (1-3)")
    argparser.add_argument("--base_dir", type=str, required=True, help="Base directory containing folders")
    Nth = argparser.parse_args().node
    if Nth < 1 or Nth > 3:
        raise ValueError("Node number must be between 1 and 3")
    baseDir = argparser.parse_args().base_dir
    if not os.path.isdir(baseDir):
        raise ValueError(f"Base directory {baseDir} does not exist or is not a directory")
    allFolders = getFolderList(baseDir)
    assignedFolders = [folder for i, folder in enumerate(allFolders) if (i % 3) + 1 == Nth]
    print(f"Node {Nth} processing {len(assignedFolders)} folders out of {len(allFolders)} total folders.")
    process_folders(assignedFolders)
    print(f"Node {Nth} completed processing assigned folders.")