from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='voxelize',
    ext_modules=[
        CUDAExtension('voxel_layer', [
            'src/voxelization.cpp',
            'src/scatter_points_cpu.cpp',
            'src/scatter_points_cuda.cu',
            'src/voxelization_cpu.cpp',
            'src/voxelization_cuda.cu',

        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']}),
	
    ],
    cmdclass={'build_ext': BuildExtension}
)
