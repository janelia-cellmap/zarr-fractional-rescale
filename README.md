Upscaling/downscaling of large arrays with dask 

Utilizes ndimage.zoom for spline interpolation. The input is extended by replicating the last pixel ('nearest').

How to run fractional upscaling/downscaling on an lsf cluster:

`bsub -n 2 -J rescale -o PATH_TO_JOBLOG 'umask 002; python3 fraction_upscale.py --src=PATH_TO_INPUT_ZARR_ARRAY --dest=PATH_TO_OUTPUT_ZARR_CONTAINER --cluster=lsf/local --workers=40 --ratio=1.5 --arr_name=NAME_OF_THE_OUTPUT_ZARR_ARRAY'`

Note: "ratio" parameter is the ratio of input scale(voxel size in nm) to the output scale
