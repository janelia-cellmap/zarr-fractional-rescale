from dask.distributed import Client, wait, LocalCluster
import zarr
from dask.array.core import slices_from_chunks, normalize_chunks
from numcodecs import Zstd
import math
from fractions import Fraction
from typing import Tuple  
import numpy as np
import  scipy.ndimage as ndimage
import dask.array as da
import os
from dask import config as cfg
from dask_jobqueue import LSFCluster
import click
from random import randint
from toolz import partition_all
import time
from cellmap_utils import get_multiscale_metadata
import pint
import logging 

def is_power_of_two(n):
    if n <= 0:
        return False
    log2_n = math.log2(n)
    log2_inv_n = math.log2(1/n)
    return math.isclose(log2_n, round(log2_n)) or math.isclose(log2_inv_n, round(log2_inv_n))

def get_slices(src_arr: zarr.Array,
                   slab_src: int,
                   slab_dest: int):
    slab_count_in = slab_src
    slab_count_out = slab_dest
    # check how many pixel_slabs are approximately in one chunk:
    chunk_shape = src_arr.chunks
    slices_dims = [int(dim / slab_count_in) * slab_count_in for dim in chunk_shape]

    # calculate input slices
    in_slices = slices_from_chunks(normalize_chunks(slices_dims, shape = src_arr.shape))
    # caclulate output slices
    out_slices = []
    for slice_voxel_in in in_slices:
        slice_voxel_out = []
        for sl in slice_voxel_in:
            slice_voxel_out.append(slice(int(sl.start / slab_count_in) * slab_count_out,
                                        int(sl.stop / slab_count_in) * slab_count_out,
                                        None))
        out_slices.append(tuple(slice_voxel_out))
    return (in_slices, out_slices)

def fractional_reshape(src_arr : zarr.Array,
                       dest_arr : zarr.Array,
                        slab_src: int,
                   slab_dest: int,
                       slices_map : Tuple[Tuple[slice, ...]],
                       interpolation_order : int
                      ):
    input_slices = slices_map[0]
    out_slices = slices_map[1]

    padding_width_in = slab_src
    padding_width_out = slab_dest
    
    # construct input/output array 
    src_sl_shape = tuple( (sl.stop - sl.start) + padding_width_in * 2 for sl in input_slices)
    dest_sl_shape = tuple( (sl.stop - sl.start) + padding_width_out * 2 for sl in out_slices)
    src_data = np.empty(shape= src_sl_shape, dtype=src_arr.dtype)
    
    #upsampling
    block_start_idxs= [False if sl.start==0 or sl.stop==sh else True for sl, sh in zip(input_slices, src_arr.shape)]
    if all(block_start_idxs):
        src_slices = tuple(slice(sl.start - padding_width_in, sl.stop + padding_width_in, None) for sl in input_slices)
        src_data = src_arr[src_slices]
        zoomed_data = ndimage.zoom(src_data, (slab_dest/slab_src, )*3, order=interpolation_order, mode='nearest')
        out_data = zoomed_data[padding_width_out: -padding_width_out, padding_width_out: -padding_width_out, padding_width_out: -padding_width_out]
        dest_arr[out_slices] = out_data
    else:
        src_data = src_arr[input_slices]
        if not (src_data == 0).all():
            zoomed_data = ndimage.zoom(src_data, (slab_dest/slab_src, )*3, order=interpolation_order, mode='nearest')
            
            # Figure out in which dimension (zoomed_data shape) != (chunksize shape)
            padding = [not ((sl.stop - sl.start) == zoomed_dim) for sl,zoomed_dim in zip(out_slices, zoomed_data.shape)]
            
            # get the part of the zoomed data that has the same dimensions as out_slices
            zoomed_data_slicing = []            
            for padding_bool, out_slice, zoomed_dim in zip(padding, out_slices, zoomed_data.shape):
                out_slice_dim = out_slice.stop - out_slice.start
                if padding_bool:
                        diff = zoomed_dim - out_slice_dim
                        zoomed_data_slicing.append(slice(0, zoomed_dim - diff, None))                        
                else:
                    zoomed_data_slicing.append(slice(0, out_slice_dim, None))
                    
            out_data  = zoomed_data[tuple(zoomed_data_slicing)]
            dest_arr[out_slices] = out_data
        


@click.command()
@click.option('--src','-s',type=click.Path(exists = True), help='Input .zarr array location.')
@click.option('--dest','-d',type=click.STRING, help='Output .zarr array location.')
@click.option('--cluster', '-c', type=click.STRING, help="Dask cluster options: 'local' or 'lsf'")
@click.option('--workers','-w',default=100,type=click.INT, help = "Number of dask workers")
@click.option('--input_scale','-is',default="1" ,type=click.STRING, help = "Physical voxel size (integer) of the input array the needs to be rescaled")
@click.option('--output_scale','-os',default="1" ,type=click.STRING, help = "Physical voxel size (integer) of the output rescaled array")
@click.option('--dataset_name','-an',default="" ,type=click.STRING, help = "Name of the output array")
@click.option('--interpolation_order', '-io', default=3, type=click.INT, help="The order of the spline interpolation, default is 3. The order has to be in the range 0-5.")
@click.option('--ome_zarr', '-ome', is_flag=True, type=click.BOOL, help="Store rescaled array as an ome-ngff dataset with multiscale schema if flag is present. Otherwise, store as a zarr array")
@click.option('--dask_log_dir','-l', type=click.STRING, help="The path of the parent directory for all LSF worker logs.  Omit if you want worker logs to be emailed to you.")
def cli(src,
        dest,
        cluster,
        workers,
        input_scale,
        output_scale,
        dataset_name,
        interpolation_order,
        ome_zarr,
        dask_log_dir
        ):
    
    logging.basicConfig(level=logging.INFO, 
                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if cluster=='lsf':        
        num_cores = 1
        cluster = LSFCluster(
            cores=num_cores,
            processes=num_cores,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="48:00",
            local_directory = "/scratch/$USER/",
            log_directory=dask_log_dir
            )
    elif cluster=='local':
        cluster = LocalCluster()
    client = Client(cluster)        
    client.cluster.scale(workers)
    
    # with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
    #     text_file.write(str(client.dashboard_link))
    logging.info(f'dask dashboard link: {client.dashboard_link}')
    src_group_path, src_arr_name = os.path.split(src)

    zg = zarr.open(src_group_path, mode = 'r')
    z_arr_src = zg[src_arr_name]
    
    zs_dest = zarr.NestedDirectoryStore(dest)
    zg_dest_root = zarr.open(zs_dest, mode = 'a')
    
    if dataset_name=='':
        dataset_name = f'dataset_{randint(0, 1000)}'
        logging.info(f'Output dataset name: {dataset_name}')
    
    ureg = pint.UnitRegistry()
    input = ureg.Quantity(input_scale)
    output = ureg.Quantity(output_scale)
                
    ratio = Fraction(str(int(round(input.to('nanometer').magnitude)))) / Fraction(str(int(round(output.to('nanometer').magnitude))))
    slab_dest = ratio.numerator
    slab_src = ratio.denominator
    
    # # calculate destination array shape and chunks:
    # # when downsampling/upsampling by a factor of 2^n:
    if is_power_of_two(ratio):
        dest_chunks = z_arr_src.chunks
    else:
        # TODO: proper chunkshape for upsampling, otherwise it would scale chunks by 'ratio' factor
        dest_chunks = [int(dim / slab_src) * slab_dest for dim in z_arr_src.chunks]
    
    in_slices, out_slices = get_slices(z_arr_src, slab_src, slab_dest)
    dest_shape = tuple(dim_slice.stop for dim_slice in out_slices[-1])
    
    logging.info(f'rescaled array chunk shape: {dest_chunks}')
    logging.info(f'rescaled array shape: {dest_shape}')
    
    if ome_zarr:
        arr_name  = 's0'
        zg_dest_root.require_group(dataset_name, overwrite=False)
        zg_dest = zg_dest_root[dataset_name]
        
        #write ome-ngff metadata into .zattrs in the dataset group
        zg_dest.attrs['multiscales'] = get_multiscale_metadata([float(output.magnitude),]*3, [0.0,]*3, 0, str(output.units), name=dataset_name)['multiscales']
    
    else:
        arr_name = dataset_name
        zg_dest = zg_dest_root
        
    z_arr_dest = zg_dest.require_dataset(arr_name,
                                        shape=dest_shape, 
                                        dtype=z_arr_src.dtype, 
                                        chunks=dest_chunks, 
                                        compressor=Zstd(level=6),
                                        fill_value=0,
                                        exact=True)
    
    #break the slices up into batches, to make things easier for the dask scheduler
    out_slices_partitioned = tuple(partition_all(100000, list(zip(in_slices, out_slices))))
    for idx, part in enumerate(out_slices_partitioned):
        start = time.time()
        
        futures = client.map(lambda x: fractional_reshape(z_arr_src, z_arr_dest, slab_src,  slab_dest, x, interpolation_order), part)
        result = wait(futures)
        
        logging.info(f'Completed {len(part)} tasks in {time.time() - start}s')

if __name__ == '__main__':
    cli()
