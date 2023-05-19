import sys
import os


def get_scripts(*args):
    contents = {}
    for path in args:
        with open(path, 'r') as file:
            contents[path] = file.read()
    return contents
