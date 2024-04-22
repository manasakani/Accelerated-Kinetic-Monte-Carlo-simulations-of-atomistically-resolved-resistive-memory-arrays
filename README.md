# DeviceKMC

KMC simulation code for VCM RRAM. Models the time evolution of conductive filaments in atomically-resolved filamentary RRAM devices under varying potential and thermal gradients. Currently set up to model a TiN-HfO2/Ti-TiN stack. 

This repository contains the code used for the submitted manuscript titled "Accelerated Kinetic Monte Carlo simulations of atomistically-resoled resistive memory arrays". 

There are two example simulations made available:

1. A 5 nm device, corresponding to the structure shown in Fig. 2a, available in /structures/5nm_device. The runtime for this example is low (roughly 10 seconds)

2. A 40 nm crossbar, corresponding to the structure shown in Fig. 2c, available in /structures/40nm_crossbar. The runtime for this example is very long as it requires an initialization step (20min on one node).

The inputs to the code are (1) a parameters.txt file, and (2) a structure file (which is named in the line 'restart_file' within the parameters.txt). Both of these are available for the two examples (5nm_device and 40nm_crossbar).


Environment:

The code was compiled and run on the LUMI Supercomputer. All environment variables are specified in the run_job_lumi.sh file.

NOTE: Due to LUMI being down for one week during the Artifact Description submission, this repository has recently been updated with the latest changes and documentation.