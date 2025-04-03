from dask.distributed import Client, wait, LocalCluster
import zarr
from dask.array.core import slices_from_chunks, normalize_chunks
from numcodecs import Zstd
from math import lcm
from fractions import Fraction
from typing import Literal, Tuple, Union
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

def get_slices(src_arr: zarr.Array,
                   dest_arr: zarr.Array):
    slab_count_in = Fraction(src_arr.shape[0], dest_arr.shape[0]).numerator
    slab_count_out = Fraction(src_arr.shape[0], dest_arr.shape[0]).denominator

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
                       slices_map : Tuple[Tuple[slice, ...]]
                      ):
    input_slices = slices_map[0]
    out_slices = slices_map[1]

    padding_width_in = Fraction(src_arr.shape[0], dest_arr.shape[0]).numerator
    padding_width_out = Fraction(src_arr.shape[0], dest_arr.shape[0]).denominator
    # construct input/output array 
    
    src_shape = tuple( (sl.stop - sl.start) + padding_width_in * 2 for sl in input_slices)
    dest_shape = tuple( (sl.stop - sl.start) + padding_width_out * 2 for sl in out_slices)
    src_data = np.empty(shape= src_shape, dtype=src_arr.dtype)
    dest_data = np.empty(shape= dest_shape, dtype= dest_arr.dtype)   
    
    #upsampling
    block_start_idxs= [False if sl.start==0 or sl.stop==sh else True for sl, sh in zip(input_slices, src_arr.shape)]
    if all(block_start_idxs):
        src_slices = tuple(slice(sl.start - padding_width_in, sl.stop + padding_width_in, None) for sl in input_slices)
        src_data = src_arr[src_slices]
        zoomed_data = ndimage.zoom(src_data, (padding_width_out/padding_width_in, )*3, order=0, mode='nearest')
        out_data = zoomed_data[padding_width_out: -padding_width_out, padding_width_out: -padding_width_out, padding_width_out: -padding_width_out]
        dest_arr[out_slices] = out_data
    else:
        src_data = src_arr[input_slices]
        zoomed_data = ndimage.zoom(src_data, (padding_width_out/padding_width_in, )*3, order=0, mode='nearest')
        dest_arr[out_slices] = zoomed_data




@click.command()
@click.option('--src','-s',type=click.Path(exists = True), help='Input .zarr array location.')
@click.option('--dest','-d',type=click.STRING, help='Output .zarr array location.')
@click.option('--cluster', '-c', type=click.STRING, help="Dask cluster options: 'local' or 'lsf'")
@click.option('--workers','-w',default=100,type=click.INT, help = "Number of dask workers")
@click.option('--ratio','-w',default="1" ,type=click.STRING, help = "Ratio of input scale to the output scale")
@click.option('--arr_name','-an',default="" ,type=click.STRING, help = "Name of the output array")
def cli(src, dest, cluster, workers, ratio, arr_name):
    if cluster=='lsf':
        # cfg.set({'distributed.scheduler.worker-ttl': None})
        # cfg.set({"distributed.comm.retry.count": 10})
        # cfg.set({"distributed.comm.timeouts.connect": 30})
        # cfg.set({"distributed.worker.memory.terminate": False})
        # cfg.set({'distributed.scheduler.allowed-failures': 100})
        
        num_cores = 1
        cluster = LSFCluster(
            cores=num_cores,
            processes=num_cores,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="48:00",
            local_directory = "/scratch/$USER/"
            )
    elif cluster=='local':
        cluster = LocalCluster()
    client = Client(cluster)
        
    client.cluster.scale(workers)
    
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "a") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)
    src_group_path, src_arr_name = os.path.split(src)
    zs = zarr.NestedDirectoryStore(src_group_path)
    zg = zarr.open(zs, mode = 'r')
    z_arr_src = zg[src_arr_name]
    
    
    zs_dest = zarr.NestedDirectoryStore(dest)
    zg_dest = zarr.open(zs_dest, mode = 'a')
    
    
    if arr_name=='':
        arr_name = f'arr_{randint(0, 1000)}'
        print(f'Output array name: {arr_name}')
    
    
    
    slab_dest = Fraction(ratio).numerator
    slab_src = Fraction(ratio).denominator
    # check how many voxel slabs fit along every dimension of a source array
    src_arr_slab_count = [int(dim / slab_src) for dim in z_arr_src.shape]
    
    # calculate destination array shape and chunks:
    dest_shape = [slab_dest *  slab_count_dim for slab_count_dim in  src_arr_slab_count] 
    dest_chunks = [int(dim / slab_src) * slab_dest for dim in z_arr_src.chunks]
    print(dest_chunks)
    print(dest_shape)
    dest_shape = tuple(int(dim*float(Fraction(ratio))) for dim in z_arr_src.shape)
    z_arr_dest = zg_dest.require_dataset(arr_name,
                                        shape=dest_shape, 
                                        dtype=z_arr_src.dtype, 
                                        chunks=dest_chunks, 
                                        compressor=Zstd(level=6),
                                        fill_value=0,
                                        exact=True)
    ratio = [src_dim/dest_dim for src_dim, dest_dim in zip(z_arr_src.shape, z_arr_dest.shape)]
    print(f'Ratio output/input: {ratio}')

    in_slices, out_slices = get_slices(z_arr_src, z_arr_dest)

    #break the slices up into batches, to make things easier for the dask scheduler
    out_slices_partitioned = tuple(partition_all(100000, list(zip(in_slices, out_slices))))
    for idx, part in enumerate(out_slices_partitioned):
        print(f'{idx + 1} / {len(out_slices_partitioned)}')
        start = time.time()
        
        futures = client.map(lambda x: fractional_reshape(z_arr_src, z_arr_dest, x), part)
        result = wait(futures)
        
        print(f'Completed {len(part)} tasks in {time.time() - start}s')
    
if __name__ == '__main__':
    cli()
