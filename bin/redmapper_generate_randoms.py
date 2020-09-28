#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate redmapper randoms for input to zmask code')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-n', '--nrands', action='store', type=int, required=True,
                        help='Number of randoms to generate')

    parser.add_argument('-m', '--mask', action='store', type=int,
                        required=True,
                        help='Apply geometric mask to randoms')

    args = parser.parse_args()

    config = redmapper.Configuration(args.configfile)

    generateRandoms = redmapper.GenerateRandoms(config,
                                                use_geometry_mask=args.mask)
    generateRandoms.generate_randoms(args.nrands)

