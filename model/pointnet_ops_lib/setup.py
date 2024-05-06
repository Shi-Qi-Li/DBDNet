import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

src = 'src'
sources = [os.path.join(root, file) for root, dirs, files in os.walk(src)
           for file in files
           if file.endswith('.cpp') or file.endswith('.cu')]

os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5;8.0;8.6"
setup(
    name='pointnet_ops',
    version='1.0',
    install_requires=["torch", "numpy"],
    packages=["pointnet_ops"],
    package_dir={"pointnet_ops": "functions"},
    ext_modules=[
        CUDAExtension(
            name='pointops._C',
            sources=sources,
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
