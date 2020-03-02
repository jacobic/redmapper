#!/usr/bin/env python
"""
Create a batch configuration script to submit to a cluster.
"""

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import yaml
import healpy as hp
import numpy as np
import glob

import redmapper

def create_batchconfig(filename):
    with open(filename, 'w') as f:
        f.write("""
batchname:
   setup: ''
   batch: 'lsf'
   requirements: ''
""")

def load_batchconfig(filename):
    """
    Load a batch configuration file.

    Parameters
    ----------
    filename: `str`
       Filename of batch configuration file

    Returns
    -------
    yaml_data: `dict`
       Dict of parameters from configuration file.
    """
    with open(filename) as f:
        yaml_data = yaml.load(f, Loader=yaml.SafeLoader)

    for key in yaml_data.keys():
        if 'batch' not in yaml_data[key]:
            raise ValueError("Missing 'batch' key for %s section in %s." % (key, filename))
        if 'setup' not in yaml_data[key]:
            yaml_data[key]['setup'] = ''
        if 'requirements' not in yaml_data[key]:
            yaml_data[key]['requirements'] = ''
        if 'taskfarmer' not in yaml_data[key]:
            yaml_data[key]['taskfarmer'] = False
        if 'image' not in yaml_data[key]:
            yaml_data[key]['image'] = ''
        if 'constraint' not in yaml_data[key]:
            yaml_data[key]['constraint'] = ''
        if 'qos' not in yaml_data[key]:
            yaml_data[key]['qos'] = ''
        if 'cpus_per_node' not in yaml_data[key]:
            yaml_data[key]['cpus_per_node'] = -1
        if 'mem_per_node' not in yaml_data[key]:
            yaml_data[key]['mem_per_node'] = 0.0

    return yaml_data


batchconfigfile = os.path.join(os.environ['HOME'], '.redmapper_batch.yml')
if not os.path.isfile(batchconfigfile):
    create_batchconfig(batchconfigfile)
    print("Please edit %s with batch configuration and rerun." % (batchconfigfile))

batchconfig = load_batchconfig(batchconfigfile)

if len(batchconfig) > 1:
    mode_required = True
else:
    mode_required = False

parser = argparse.ArgumentParser(description="Create a batch file for running redmapper codes")

parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                    help='YAML config file')
parser.add_argument('-r', '--runmode', action='store', type=int, required=True,
                    help='Run mode.  0 is full finder run.  1 is zred run.')
parser.add_argument('-b', '--batchmode', action='store', type=str, required=mode_required,
                    help='Batch mode, defined in ~/.redmapper_batch.yml')
parser.add_argument('-w', '--walltime', action='store', type=int, required=False,
                    help='Wall time (override default)')
parser.add_argument('-n', '--nside', action='store', type=int, required=False,
                    help='Parallelization nside (optional, can use default)')
parser.add_argument('-N', '--nodes', action='store', type=int, required=False,
                    default=2, help='Number of nodes to run (for nersc)')

args = parser.parse_args()

if not mode_required and args.batchmode is None:
    batchmode = list(batchconfig.keys())[0]
else:
    batchmode = args.batchmode

# Read in the config file

config = redmapper.Configuration(args.configfile)

if len(config.hpix) != 0:
    raise ValueError("Cannot run redmapper in batch mode with hpix not an empty list (full sky)")

# Check the nside

nside = args.nside

if args.runmode == 0:
    # This is a full run
    if nside is None:
        nside = 4
    # nside = config.nside_batch_run
    jobtype = 'run'
    default_walltime = 72*60
    memory = 4000
elif args.runmode == 1:
    # This is a zred run
    if nside is None:
        nside = 8
    jobtype = 'zred'
    default_walltime = 5*60
    memory = 2000
else:
    raise RuntimeError("Unsupported runmode: %d" % (args.runmode))

if args.walltime is None:
    walltime = default_walltime
else:
    walltime = args.walltime

jobname = '%s_%s' % (config.outbase, jobtype)

# Determine which pixels overlap the galaxy file...

tab = redmapper.Entry.from_fits_file(config.galfile)

theta, phi = hp.pix2ang(tab.nside, tab.hpix)
hpix_run = np.unique(hp.ang2pix(nside, theta, phi))

# Make the batch script in a "jobs" directory

cwd = os.getcwd()
jobpath = os.path.join(cwd, 'jobs')

if not os.path.isdir(jobpath):
    os.makedirs(jobpath)

# Will want to check for previous (failed) jobs

test = glob.glob(os.path.join(jobpath, '%s_?.job' % (jobname)))
index = len(test)

if args.runmode == 0:
    # Run in the directory where the config file is, by default
    run_command = 'redmapper_run_redmapper_pixel.py -c %s -p %%s -n %d -d %s' % (
        (os.path.abspath(args.configfile),
         nside,
         os.path.dirname(os.path.abspath(args.configfile))))
elif args.runmode == 1:
    run_command = 'redmapper_run_zred_pixel.py -c %s -p %%s -n %d -d %s' % (
        (os.path.abspath(args.configfile),
         nside,
         os.path.dirname(os.path.abspath(args.configfile))))

jobfile = os.path.join(jobpath, '%s_%d.job' % (jobname, index + 1))

with open(jobfile, 'w') as jf:
    write_jobarray = True
    if (batchconfig[batchmode]['batch'] == 'lsf'):
        # LSF mode
        jf.write("#BSUB -R '%s'\n" % (batchconfig[batchmode]['requirements']))
        jf.write("#BSUB -R 'rusage[mem=%d]'\n" % (memory))
        jf.write("#BSUB -J %s[1-%d]\n" % (jobname, hpix_run.size))
        jf.write("#BSUB -oo %s\n" % (os.path.join(jobpath, '%s_%%J_%%I.log' % (jobname))))
        jf.write("#BSUB -n 1\n")
        jf.write("#BSUB -W %d\n\n" % (walltime))

        index_string = '${pixarr[LSB_JOBINDEX-1]}'

    elif (batchconfig[batchmode]['batch'] == 'pbs'):
        # PBS mode
        ppn = batchconfig[batchmode]['ppn']
        n_nodes = int(np.ceil(float(hpix_run.size) / float(ppn)))
        jf.write("#PBS -q %s\n" % (batchconfig[batchmode]['queue']))
        jf.write("#PBS -l nodes=%d:ppn=%d\n" % (n_nodes, ppn))
        jf.write("#PBS -l walltime=%d:00:00\n" % (int(walltime / 60)))
        jf.write("#PBS -l mem=%dmb\n" % (memory))
        jf.write("#PBS -j oe\n")
        jf.write('N_CPU=%d\n' % (n_nodes * batchconfig[batchmode]['ppn']))

    elif (batchconfig[batchmode]['batch'] == 'slurm'):
        # SLURM mode
        ppn = batchconfig[batchmode]['ppn']
        n_nodes = int(np.ceil(float(hpix_run.size) / float(ppn)))
        jf.write("#SBATCH -p %s\n" % (batchconfig[batchmode]['queue']))
        jf.write("#SBATCH -N s=%d\n" % (n_nodes))
        jf.write("#SBATCH --ntasks-per-node %d\n" % (ppn))
        jf.write("#SBATCH -t=%d:00:00\n" % (int(walltime / 60)))
        jf.write("#SBATCH --mem %dmb\n" % (memory/n_nodes))
        jf.write("#SBATCH -J %s[1-%d]\n" % (jobname, hpix_run.size))
    else:
        # Nothing else supported
        raise RuntimeError("Only LSF, PBS and SLURM supported at this time.")

    if write_jobarray:
        jf.write("pixarr=(")
        for hpix in hpix_run:
            jf.write("%d " % (hpix))
        jf.write(")\n\n")

        jf.write("%s\n\n" % (batchconfig[batchmode]['setup']))

        cmd = run_command % (index_string)
        jf.write("%s\n" % (cmd))
