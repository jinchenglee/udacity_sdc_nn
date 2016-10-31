#!/usr/bin/env python3

import tables
import os
import numpy as np

# Define compresison filter
FILTERS = tables.Filters(complib='zlib',complevel=5)

for file_num in range(490):
    # Copy and unzip
    os.system("cp /home/vitob/Downloads/deepdrive_hdf5/train_"+str(file_num).zfill(4)+".h5.gz /home/vitob/git_projects/udacity_sdc_nn/train.h5.gz")
    os.system("cd /home/vitob/git_projects/udacity_sdc_nn;gzip -d train.h5.gz")

    # Open source and destination
    f  = tables.open_file("./train.h5",'r')
    fc = tables.open_file("./train_"+str(file_num).zfill(4)+".zlib.h5",'w',filters=FILTERS)
    
    # Conversion, or "copy with compression"
    fc.create_earray('/','images',obj=np.array(f.root.images))
    fc.create_earray('/','targets',obj=np.array(f.root.targets))
    fc.create_earray('/','vehicle_states',obj=np.array(f.root.vehicle_states))
    
    # Close files
    f.close()
    fc.close()

    # Delete local file
    os.system("rm -fr /home/vitob/git_projects/udacity_sdc_nn/train.h5")
