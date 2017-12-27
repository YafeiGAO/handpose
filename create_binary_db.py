#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
    Script to convert Rendered Handpose Dataset into binary files,
    which allows for much faster reading than plain image files.

    Set "path_to_db" and "set" accordingly.

    In order to use this file you need to download and unzip the dataset first.
"""
from __future__ import print_function, unicode_literals

import pickle
import os
import scipy.misc
import struct
import scipy.io#yafei.@@

# SET THIS to where RHD is located on your machine
path_to_db = '/home/alien/Yafei/deep-prior-pp/data/New_nyu/'#yafei.@@
#path_to_db = '/media/alien/SAMSUNG/dataset/'

# chose if you want to create a binary for training or evaluation set
#set = 'train3d'
set = 'test3d'

### No more changes below this line ###


# function to write the binary file
def write_to_binary(file_handle, image, depth, mask, kp_coord_xyz, kp_coord_uv, kp_visible,kp_depth):
    """" Writes records to an open binary file. """
    bytes_written = 0
    # 1. write kp_coord_xyz
    for coord in kp_coord_xyz:
        file_handle.write(struct.pack('f', coord[0]))
        file_handle.write(struct.pack('f', coord[1]))
        file_handle.write(struct.pack('f', coord[2]))
    bytes_written += 4*kp_coord_xyz.shape[0]*kp_coord_xyz.shape[1]

    # 2. write kp_coord_uv
    for coord in kp_coord_uv:
        file_handle.write(struct.pack('f', coord[0]))
        file_handle.write(struct.pack('f', coord[1]))
    bytes_written += 4*kp_coord_uv.shape[0]*kp_coord_uv.shape[1]

    #3. write K_mat
    for K_row in K_mat:
        for K_element in K_row:
            file_handle.write(struct.pack('f', K_element))
    bytes_written += 4*9

    file_handle.write(struct.pack('B', 255))
    file_handle.write(struct.pack('B', 255))
    bytes_written += 2

    # 4. write image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            file_handle.write(struct.pack('B', image[x, y, 0]))
            file_handle.write(struct.pack('B', image[x, y, 1]))
            file_handle.write(struct.pack('B', image[x, y, 2]))
    bytes_written += 4*image.shape[0]*image.shape[1]*image.shape[2]

    # 5. write depth
    for x in range(depth.shape[0]):
        for y in range(image.shape[1]):
            file_handle.write(struct.pack('B', depth[x, y, 0]))
            file_handle.write(struct.pack('B', depth[x, y, 1]))
            file_handle.write(struct.pack('B', depth[x, y, 2]))
    bytes_written += 4*depth.shape[0]*depth.shape[1]*depth.shape[2]

    # 6. write mask
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            file_handle.write(struct.pack('B', mask[x, y]))
    bytes_written += 4*mask.shape[0]*mask.shape[1]

    # 7. write visibility
    for x in range(kp_visible.shape[0]):
        file_handle.write(struct.pack('B', kp_visible[x]))
    bytes_written += kp_visible.shape[0]

    # 8. write depth
    for x in range(kp_depth.shape[0]):
        file_handle.write(struct.pack('f', kp_depth[x]))
    bytes_written += 4*kp_depth.shape[0]
    #print('bytes_written', bytes_written)

# binary file we will write
file_name_out = './data/bin/nyu_%s_1.bin' % set #full size:-> ubu; 1: ->1;...

if not os.path.exists('./data/bin'):
    os.mkdir('./data/bin')

# iterate samples of the set and write to binary file
with open(file_name_out, 'wb') as fo: 
    print (path_to_db+set+'/joint_data.mat')
    mat = scipy.io.loadmat(path_to_db+set+'/joint_data.mat')
    num = mat['joint_xyz'][0]
    num_samples = num.shape[0]
    print('num of sampels: %d'%num_samples)
    count = 1
    for i in range(0,1):
        print('ROUND: %d' %(i+1))
        #names = mat['joint_names'][i] #<1 #[0]->1;[1]->2;[2]->3
        joints3D = mat['joint_xyz'][i] #<3
        joints2D = mat['joint_uvd'][i]
        #print (joints3D[8251][35][2])  #<8252,<36,<3
        #print (names[35]) #<36
        #print (joints2D[8251][35][2])  #<8252,<36,<3
        #rg = num_samples + 1
        for sample_id in range(1,num_samples):
            # load data
            s = "%07d" % sample_id
            image = scipy.misc.imread(os.path.join(path_to_db, set, 'color', '%s_%s.png' %(i+1,s)))#1->1
            depth = scipy.misc.imread(os.path.join(path_to_db,set,'depth', '%s_%s.png' %(i+1,s)))

            mask = scipy.misc.imread(os.path.join(path_to_db, set, 'mask', '%s_%s.png' %(i+1,s)))
	    
            #print(path_to_db, set, 'mask', '%s_%s.png' %(i+1,s))
            kp_coord_uv = joints2D[sample_id,:,0:2]
            kp_coord_xyz = joints3D[sample_id]
            kp_visible = joints2D[sample_id,:,2] == 1
            kp_depth = joints2D[sample_id,:,3]
            #print(kp_coord_xyz.shape)
            #print(kp_coord_uv.shape)
            #print(sample_id)
            count += 1
            if (sample_id % 100) == 0:
                print('%d / %d images done: %.3f percent' % (sample_id, num_samples, sample_id*100.0/num_samples))
            write_to_binary(fo, image,depth, mask, kp_coord_xyz, kp_coord_uv, kp_visible,kp_depth)

        print('recorded: %d images' %count)
    print(kp_coord_uv.shape)
