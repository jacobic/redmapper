#!/usr/bin/env python

import os
import sys
import subprocess
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Run multiple redmapper pixels on the same node')

parser.add_argument('-c', '--command', action='store', type=str, required=True, help='Command to run')
parser.add_argument('-P', '--pixels', action='store', type=str, required=True, help='Comma-separated list of pixels')

args = parser.parse_args()

pixels = args.pixels.split(',')

class RunCommand(object):
    def __init__(self, command):
        self.command = command

    def __call__(self, pixel):
        full_command = self.command + ' -p ' + pixel
        print(full_command)
        subprocess.call(full_command, shell=True)

runCommand = RunCommand(args.command)

pool = multiprocessing.Pool(processes=len(pixels))
results = []

for pixel in pixels:
    results.append(pool.apply_async(runCommand, (pixel, )))

pool.close()
pool.join()

for res in results:
    res.get()
