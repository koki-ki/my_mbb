# coding: utf-8

import os
import shutil

PATH_FROM_IMPORT = "/Users/ccilab/Desktop/BK/20200121/"  # Path to directory of exp_grid_ds or exp_grid_lr
PATH_TO_IMPORT = "../res"
IGNORE = [".DS_Store"]

def find_base_directory_path(from_path, to_path):
    to_path_basename = os.path.basename(to_path)
    from_path_conts = from_path.split("/")
    position = -1
    for i, fpc in enumerate(from_path_conts):
        if fpc == to_path_basename:
            position = i
            break
    if position >= 0:
        new_path = os.path.join(to_path, "/".join(from_path_conts[(i+1):]))
        return new_path
    else:
        print("SKIP: Could not find base directory: place=%s, searching=%s"
              % (from_path, to_path_basename))
        return ""

def update_copy(from_path, to_path):
    if os.path.isfile(to_path):
        from_path_time = os.stat(from_path).st_mtime
        to_path_time = os.stat(to_path).st_mtime
        if from_path_time > to_path_time:
            print("UPDATE COPY: %s -> %s" % (from_path, to_path))
            shutil.copy(from_path, to_path)
        else:
            print("SKIP COPY: %s -> %s" % (from_path, to_path))
    else:
        print("NEW COPY: %s -> %s" % (from_path, to_path))
        shutil.copy(from_path, to_path)

for cur_dir, dirs, files in os.walk(PATH_FROM_IMPORT):
    new_dir = find_base_directory_path(cur_dir, PATH_TO_IMPORT)
    if new_dir == "":
        continue
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    for fs in files:
        if fs in IGNORE:
            continue
        old_path = os.path.join(cur_dir, fs)
        new_path = os.path.join(new_dir, fs)
        update_copy(old_path, new_path)
