'''
    Copyright (C) 2014-2019, Johannes Pekkilae, Miikka Vaeisalae.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
'''

# This module is for reading data.

import numpy as np


def set_dtype(endian, AcRealSize):
    if endian == 0:
        en = '>'
    elif endian == 1: 
        en = '<'
    type_instruction = en + 'f' + str(AcRealSize)
    print("type_instruction", type_instruction)
    my_dtype = np.dtype(type_instruction)
    return my_dtype

def read_bin(fname, fdir, fnum, minfo, numtype=np.longdouble):
    '''Read in a floating point array'''
    filename = fdir + fname + '_' + fnum + '.mesh'
    datas = np.DataSource()
    read_ok = datas.exists(filename)

    my_dtype = set_dtype(minfo.contents['endian'], minfo.contents['AcRealSize'])

    if read_ok:
        print(filename)
        array = np.fromfile(filename, dtype=my_dtype)

        timestamp = array[0]

        array = np.reshape(array[1:], (minfo.contents['AC_mx'], 
                                       minfo.contents['AC_my'], 
                                       minfo.contents['AC_mz']), order='F')
    else:
        array = None
        timestamp = None
     
    return array, timestamp, read_ok 

def read_meshtxt(fdir, fname):
    with open(fdir+fname) as f:
        filetext = f.read().splitlines()

    contents = {}  

    for line in filetext:
        line = line.split()
        if line[0] == 'int':
            contents[line[1]] = np.int(line[2])
            print(line[1], contents[line[1]])
        elif line[0] == 'size_t':
            contents[line[1]] = np.int(line[2])
            print(line[1], contents[line[1]])
        elif line[0] == 'int3':
            contents[line[1]] = [np.int(line[2]), np.int(line[3]), np.int(line[4])]
            print(line[1], contents[line[1]])
        elif line[0] == 'real':
            contents[line[1]] = np.float(line[2])
            print(line[1], contents[line[1]])
        elif line[0] == 'real3':
            contents[line[1]] = [np.float(line[2]), np.float(line[3]), np.float(line[4])]
            print(line[1], contents[line[1]])
        else: 
            print(line)
            print('ERROR: ' + line[0] +' not recognized!')

    return contents

#Now just 2nd order
def DERX(array, dx):
    output = np.zeros_like(array)
    for i in range(3, array.shape[0]-3): #Keep boundary poits as 0
        #output[i,:,:] = (- array[i-1,:,:] + array[i+1,:,:])/(2.0*dx)
        output[i,:,:] =( -45.0*array[i-1,:,:] + 45.0*array[i+1,:,:]
                         + 9.0*array[i-2,:,:] -  9.0*array[i+2,:,:]
                         -     array[i-3,:,:] +      array[i+3,:,:] )/(60.0*dx)
    return output

def DERY(array, dy):
    output = np.zeros_like(array)
    for i in range(3,array.shape[1]-3):
        #output[:,i,:] = (- array[:,i-1,:] + array[:,i+1,:])/(2.0*dy)
        output[:,i,:] =( -45.0*array[:,i-1,:] + 45.0*array[:,i+1,:]
                         + 9.0*array[:,i-2,:] -  9.0*array[:,i+2,:]
                         -     array[:,i-3,:] +      array[:,i+3,:] )/(60.0*dy)
    return output

def DERZ(array, dz):
    output = np.zeros_like(array)
    for i in range(3, array.shape[2]-3):
        #output[:,:,i] = (- array[:,:,i-1] + array[:,:,i+1])/(2.0*dz)
        output[:,:,i] =( -45.0*array[:,:,i-1] + 45.0*array[:,:,i+1]
                         + 9.0*array[:,:,i-2] -  9.0*array[:,:,i+2]
                         -     array[:,:,i-3] +      array[:,:,i+3] )/(60.0*dz)
    return output

def curl(aa, minfo):
    dx = minfo.contents['AC_dsx']
    dy = minfo.contents['AC_dsy']
    dz = minfo.contents['AC_dsz']
    return (DERY(aa[2], dy)-DERZ(aa[1], dz), 
            DERZ(aa[0], dz)-DERX(aa[2], dx), 
            DERX(aa[1], dx)-DERY(aa[0], dy))


class MeshInfo():
    '''Object that contains all mesh info'''

    def __init__(self, fdir):
        self.contents = read_meshtxt(fdir, 'mesh_info.list') 

class Mesh:
    '''Class tha contains all 3d mesh data'''

    def __init__(self, fnum, fdir=""):
        fnum = str(fnum)
        self.framenum = fnum.zfill(10)

        self.minfo = MeshInfo(fdir)

        self.lnrho, self.timestamp, self.ok = read_bin('VTXBUF_LNRHO', fdir, fnum, self.minfo)

        if self.ok:

            self.ss, timestamp, ok = read_bin('VTXBUF_ENTROPY', fdir, fnum, self.minfo)

            self.accretion, timestamp, ok = read_bin('VTXBUF_ACCRETION', fdir, fnum, self.minfo)
 
            #TODO Generalize is a dict. Do not hardcode!  
            uux, timestamp, ok = read_bin('VTXBUF_UUX', fdir, fnum, self.minfo)
            uuy, timestamp, ok = read_bin('VTXBUF_UUY', fdir, fnum, self.minfo) 
            uuz, timestamp, ok = read_bin('VTXBUF_UUZ', fdir, fnum, self.minfo)
            self.uu = (uux, uuy, uuz)
            uux = []
            uuy = [] 
            uuz = []
 
            aax, timestamp, ok = read_bin('VTXBUF_AX', fdir, fnum, self.minfo)
            aay, timestamp, ok = read_bin('VTXBUF_AY', fdir, fnum, self.minfo) 
            aaz, timestamp, ok = read_bin('VTXBUF_AZ', fdir, fnum, self.minfo)
            self.aa = (aax, aay, aaz)
            aax = []
            aay = [] 
            aaz = []

             
            #self.aa[0][:,:,:] = 0.0
            #self.aa[1][:,:,:] = 0.0 
            #self.aa[2][:,:,:] = 0.0 
            #for i in range(0, self.aa[0].shape[0]):
            #    self.aa[0][:,i,:] = float(i)
   

            self.xx =  np.arange(self.minfo.contents['AC_mx']) * self.minfo.contents['AC_dsx']
            self.yy =  np.arange(self.minfo.contents['AC_my']) * self.minfo.contents['AC_dsy']
            self.zz =  np.arange(self.minfo.contents['AC_mz']) * self.minfo.contents['AC_dsz']

            self.xmid = int(self.minfo.contents['AC_mx']/2)
            self.ymid = int(self.minfo.contents['AC_my']/2)
            self.zmid = int(self.minfo.contents['AC_mz']/2)
    def Bfield(self):
        self.bb = curl(self.aa, self.minfo) 
        print(self.bb[2]) 


def parse_ts(fdir, fname):
    with open(fdir+fname) as f:
        filetext = f.read().splitlines()

    var = {}  

    line = filetext[0].split()
    for i in range(len(line)):
        line[i] = line[i].replace('VTXBUF_', "")
        line[i] = line[i].replace('UU', "uu")
        line[i] = line[i].replace('_total', "tot")
        line[i] = line[i].replace('ACCRETION', "acc")
        line[i] = line[i].replace('A', "aa")
        line[i] = line[i].replace('LNRHO', "lnrho")
        line[i] = line[i].replace('ENTROPY', "ss")
        line[i] = line[i].replace('X', "x")
        line[i] = line[i].replace('Y', "y")
        line[i] = line[i].replace('Z', "z")

    tsdata = np.loadtxt(fdir+fname,skiprows=1)

    for i in range(len(line)):
        var[line[i]] = tsdata[:,i]

    var['step'] = np.int64(var['step'])

    print("HERE ARE ALL KEYS FOR TS DATA:")
    print(var.keys())
   
    return var

class TimeSeries:
    '''Class for time series data'''

    def __init__(self, fdir="", fname="timeseries.ts"):

        self.var = parse_ts(fdir, fname)
