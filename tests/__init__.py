import sys
import os

file_dir = os.path.abspath(os.path.join(__file__, os.pardir))
root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
sys.path.append(os.path.join(root_dir, 'src'))
