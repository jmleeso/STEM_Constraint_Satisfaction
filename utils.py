import os
import json
import zipfile
import numpy as np
from typing import List

def load_memmap_from_npz(path, name):
    '''
    Can't load npy files from npz files as memap.
    Temporary work around from https://github.com/numpy/numpy/issues/5976.
    '''
    zf = zipfile.ZipFile(path)
    info = zf.NameToInfo[name + '.npy']
    assert info.compress_type == 0
    offset = zf.open(name + '.npy')._orig_compress_start

    fp = open(path, 'rb')
    fp.seek(offset)
    version = np.lib.format.read_magic(fp)
    assert version in [(1,0), (2,0)]
    if version == (1,0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2,0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    data_offset = fp.tell() # file position will be left at beginning of data
    return np.memmap(path, dtype=dtype, shape=shape,
                     order='F' if fortran_order else 'C', mode='r',
                     offset=data_offset)

def parse_file_path(file_path:str):
    directory, name_with_ext = os.path.split(file_path)
    name, extension = os.path.splitext(name_with_ext)
    
    known_extensions = ['.npy', '.ndata1']
    
    if extension not in known_extensions:
        name = name_with_ext 
        
        if os.path.exists(file_path + '.npy'):
            extension = '.npy'
        elif os.path.exists(file_path + '.ndata1'):
            extension = '.ndata1'
        elif os.path.isdir(file_path):
            extension = '' 
        else:
            extension = '' 
    
    return directory, name, extension

def collect_swift_file(file_path:str):
    directory, name, file_extension = parse_file_path(file_path)
    base_file_path = os.path.join(directory, name)
    
    if file_extension == '.npy':
        with open(base_file_path+'.json') as f: meta = json.load(f)
        data = np.load(base_file_path+'.npy', mmap_mode='r')
    elif file_extension == '.ndata1':
        file = np.load(base_file_path+'.ndata1', mmap_mode='r')
        meta = json.loads(file['metadata.json'].decode())
        data = load_memmap_from_npz(base_file_path+'.ndata1', 'data')
    elif file_extension == '':
        with open(base_file_path+'/metadata.json') as f: meta = json.load(f)
        data = np.load(base_file_path+'/data.npy', mmap_mode='r')
    else:
        raise Exception(f'The Swift files could not be collected.\n'
                         f'A file extension of `.npy` or `.ndata1` was not found.\n'
                         f'Base path was: {base_file_path} (Extension found: {file_extension})')
        
    return meta, data

class swift_json_reader:
    def __init__(self, file_path:str, signal_type:str=None, get_npy_shape:bool=True, verbose=False):
        #File handling
        self.file_path = file_path
        self.file_directory, self.file_name, self.file_extension = parse_file_path(file_path)
        self.meta, data = collect_swift_file(self.file_path)
        self.data_shape = data.shape
        self.detector = self.meta['metadata']['hardware_source'].get('source')

        #General metadata
        self.title = self.meta.get('title')

        #Axis handling
        self.is_series = True if self.meta.get('is_sequence') is not None else False
        self.is_scan = True if self.meta['metadata'].get('scan') is not None else False
        self.is_preprocessed = len(self.meta['metadata']) + len(self.meta['properties']) == 0
        self.nav_dim = self.meta['collection_dimension_count']
        self.sig_dim = self.meta['datum_dimension_count']
        self.signal_type = signal_type if signal_type is not None else self.read_signal_type()
        if verbose: print(f'Read signal type: {self.signal_type}')

        self.axes = self.read_axes_calibrations()
        if verbose: print(f'Read axes: {self.axes}')
        if get_npy_shape:
            for ax, d in zip(self.axes, self.data_shape): ax['size'] = d
        if self.is_series and self.axes[0]['units']=='': self.axes[0]['units'] = 'frame'
        
        self.infer_axes_names(verbose=verbose)
        if verbose: print(f'Built axes: {self.axes}')

        if not self.is_preprocessed:
            #Instrument metadata
            self.beam_energy = self.meta['metadata']['instrument'].get('high_tension')/1E3
            self.aberrations = self.read_abberations()

            #Scan metadata
            if self.is_scan:
                self.scan_rotation = self.meta['metadata']['scan'].get('rotation_deg')
                self.dwell_time = self.meta['metadata']['scan']['scan_device_parameters'].get('pixel_time_us') * 1E-6
            
            #Detector metadata
            self.name = self.meta['metadata']['hardware_source'].get('hardware_source_name')
            self.exposure = self.meta['metadata']['hardware_source'].get('exposure')
            if verbose: print(self.meta['metadata']['hardware_source'].get('sensor_readout_area_tlbr'))
            self.readout_area = \
                self.meta['metadata']['hardware_source']['camera_processing_parameters'].get('readout_area') if self.meta['metadata']['hardware_source'].get('camera_processing_parameters') is not None \
                else self.meta['metadata']['hardware_source'].get('sensor_readout_area_tlbr')
            self.binning = self.meta['properties'].get('binning')
            self.flip_x = self.meta.get('camera_processing_parameters').get('flip_l_r') if self.meta.get('camera_processing_parameters') is not None \
                else self.meta['properties'].get('is_flipped_horizontally')

    def read_signal_type(self):
        if self.sig_dim == 1:
            signal_type = 'EELS'
        if self.sig_dim == 2:
            sig_unit = np.asanyarray([ax['units'] for ax in self.meta['spatial_calibrations'][-2:]])
            if np.all(sig_unit == 'nm'):
                signal_type = 'Image'
            elif np.all(['rad' in u for u in sig_unit]):
                signal_type = 'diffraction'
            elif sig_unit[-1] == 'eV':
                if np.diff(self.data_shape[-2:])==0:
                    signal_type = 'diffraction'
                else:
                    signal_type = '2D-EELS'
        return signal_type

    def read_axes_calibrations(self):
        axes = [ax.copy() for ax in self.meta['spatial_calibrations']]
        return axes
    
    def infer_axes_names(self, verbose=False):
        if self.is_series: self.axes[0]['name'] = 'time'
        self.axes_rspace_dims = [i for i, ax in enumerate(self.axes) if ax['units'] in ('um','nm','A','pm')]
        for i,n in zip(self.axes_rspace_dims[::-1], 'xyz'): self.axes[i]['name'] = n 
        if self.signal_type == 'EELS':                
            self.axes_sspace_dims = [-1]
            self.axes[-1]['name'] = 'E'
        elif self.signal_type == '2D-EELS':
            self.axes_qspace_dims = [-2]
            self.axes[-2]['name'] = 'q'
            self.axes[-2]['units'] = 'px'
            self.axes_sspace_dims = [-1]
            self.axes[-1]['name'] = 'E'
        elif self.signal_type == 'diffraction':
            self.axes[-2]['name'] = 'qy'
            self.axes[-1]['name'] = 'qx'
            self.axes_qspace_dims = [-2, -1]
            for i in self.axes_qspace_dims:
                if self.detector != 'Ronchigram':
                    self.axes[i]['scale'] = 1
                    self.axes[i]['units'] = 'px'
                self.axes[i]['offset'] = -self.data_shape[i]/2 * self.axes[i]['scale']

    def read_abberations(self):
        if self.meta['metadata']['instrument'].get('ImageScanned') is None:
            return None
        aber = {k:v for k,v in self.meta['metadata']['instrument']['ImageScanned'].items() if k[0]=='C' and k[1:3].isdigit()}
        aber_c = {}
        for k,v in aber.items():
            ab = k[:3]
            if k[0]=='C' and k[-1]!='b':
                if k[-1] == 'a':
                    aber_c[ab] = aber[ab+'.a']+1j*aber[ab+'.b']
                else:
                    aber_c[ab] = v
        return aber_c

def load_swift_to_py4DSTEM(file_path:str, lazy:bool=False, verbose=False, 
                           crop_r:List[int]=None, skip_r:int=None,
                           **kwargs) -> object:
    from py4DSTEM.data import DiffractionSlice, RealSlice
    from py4DSTEM.datacube import DataCube

    # Read metadata and data
    meta = swift_json_reader(file_path, signal_type='diffraction', verbose=verbose)
    _, data = collect_swift_file(file_path)
    if crop_r is not None:
        assert len(crop_r)==meta.nav_dim
        for ax in crop_r: assert len(ax)==2
        data = data[crop_r[0][0]:crop_r[0][1], crop_r[1][0]:crop_r[1][1]]
    if skip_r is not None:
        data = data[::skip_r, ::skip_r]

    if not kwargs.get('lazy'): data = data.copy() 

    if meta.flip_x:
        if verbose: print('Detector flip_x flagged True. Reversing the qx axis.')
        data = data[...,::-1]
    else:
        if verbose: print('Detector flip_x flagged False. Not reversing the qx axis.')

    if kwargs.get('lazy'): data = data.copy() 
        
    # Determine the py4DSTEM class type
    if meta.is_series: raise Exception('Reading of series are not currently implimented. The data must be a four-dimenstional 4D-STEM scan.')
    assert len(data.shape) in (2, 4)
    if len(data.shape) == 4:
        f = DataCube(data=data)
    else:
        if meta.axes[0]['units'] == 'nm':
            f = RealSlice(data=data)
        else:
            f = DiffractionSlice(data=data)

    # Set the calibrations.
    axes = {ax['name']: ax for ax in meta.axes}
    axes = {i:axes[i] for i, k in zip(['x','y','qx','qy'], axes.keys()) if i in list(axes.keys())}
    for k,v in axes.items():
        if verbose: print(f'Storing {k} axis', v)
        if v['units'] == 'px': v['units'] = 'pixels'
        if k == 'x':
            if skip_r is None:
                xscale = v['scale']
            else:
                xscale = v['scale'] * skip_r
            f.calibration.set_R_pixel_size(xscale)
            f.calibration.set_R_pixel_units(v['units'])
        elif k == 'y':
            if v['scale'] != axes['x']['scale']:
                print("Warning: py4DSTEM currently only handles uniform x,y sampling. Setting sampling with x calibration")
        elif k == 'qx':
            f.calibration.set_Q_pixel_size(v['scale'])
            f.calibration.set_Q_pixel_units(v['units'])
            f.calibration.set_qx0_mean(-v['offset'])
        elif k == 'qy':
            f.calibration.set_qy0_mean(-v['offset'])
            if v['scale'] != axes['qx']['scale']:
                print("Warning: py4DSTEM currently only handles uniform qx,qy sampling. Setting sampling with qx calibration")
        else:
            print(f'Axes {k} is not supported and will be ignored.')
    if meta.is_scan:
        f.calibration.set_QR_rotation_degrees(meta.scan_rotation)
    
    return f