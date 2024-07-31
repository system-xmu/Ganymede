# setup()函数中变量的含义

# cuda c++混合编译实现python模块扩展
# name: 在python中，import该模块的名称
# sources: 源代码文件的名称
# laungage: 默认为c，可以改成c++
# include_dirs: 传递给gcc或者g++编译器，include的头文件目录
# library_dirs: 传递给gcc或者g++编译器，指定链接文件的目录
# libraries: 传递给gcc或者g++编译器，指定链接的文件
# extra_compile_args: 额外的gcc或者g++编译参数
# extra_links_args: 额外的传给g++或者gcc编译器的链接参数
# define_macros: 定义的宏
# undefine_macros: 取消宏

from setuptools import setup, find_packages
from setuptools.command.install import install

from subprocess import call
from typing import List
from platform import uname
from packaging import version
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, sys

this_dir = os.path.dirname(os.path.abspath(__file__))
install_dir = os.path.join(this_dir, 'install')
gpufs_install_dir = "/home/hyf/Ganymede/Comparision/gpufs"
enable_gpufs = True
if os.environ.get('DISABLE_GPUFS') == '1':
    enable_gpufs = False

libraries = ['gpufs', 'cudadevrt', 'cufft_static', 'cudart_static']
library_dirs = ['/home/hyf/Ganymede/Comparision/gpufs/lib']
sources = ['csrc/api.cpp', 'csrc/space_mgr.cpp', 'csrc/api_gpu.cu']
include_dirs = ['../gpufs/include', 'include/']
extra_objects = []
define_macros = []  # 定义的宏
ext_modules = []
cmdclass = {}


def cpp_ext_helper(name, sources, **kwargs):
    from torch.utils.cpp_extension import CppExtension
    extra_include_dirs = []
    if 'C_INCLUDE_PATH' in os.environ:
        extra_include_dirs.extend(os.environ['C_INCLUDE_PATH'].split(':'))
    if 'CPLUS_INCLUDE_PATH' in os.environ:
        extra_include_dirs.extend(os.environ['CPLUS_INCLUDE_PATH'].split(':'))
    extra_include_dirs = list(
        filter(lambda s: len(s) > 0, set(extra_include_dirs)))
    return CUDAExtension(
        name,
        [os.path.join(this_dir, path) for path in sources],
        include_dirs=[os.path.join(this_dir, 'csrc'), os.path.join(this_dir, 'include'),
                      os.path.join(gpufs_install_dir, 'include'),
                      *extra_include_dirs],
        library_dirs=[os.path.join(gpufs_install_dir, 'lib')],
        extra_compile_args = {"nvcc" : ["-dc", "-rdc=true"]},
        **kwargs
    )

def find_static_lib(lib_name: str, lib_paths: List[str] = []) -> str:
    static_lib_name = f'lib{lib_name}.a'
    lib_paths.extend(['/usr/lib', '/usr/lib64',  '/home/hyf/anaconda3/lib', ' /home/hyf/Ganymede/Comparision/gpufs/lib/']) # ,
    if os.environ.get('LIBRARY_PATH', None) is not None:
        lib_paths.extend(os.environ['LIBRARY_PATH'].split(':'))
    for lib_dir in lib_paths:
        if os.path.isdir(lib_dir):
            for filename in os.listdir(lib_dir):
                if filename == static_lib_name:
                    return os.path.join(lib_dir, static_lib_name)
    raise RuntimeError(f'{static_lib_name} is not found in {lib_paths}')


def setup_dependencies():
    build_dir = os.path.join(this_dir, 'cmake-build')
    #  if not enable_gpufs:
    #     define_macros.append(('DISABLE_GPUFS', None))
    #     sources.remove('csrc/uring.cpp')
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(install_dir, exist_ok=True)
    os.chdir(build_dir)
    call(['cmake', '..', f'-DBACKEND_INSTALL_PREFIX={install_dir}'])
    if enable_gpufs:
        # call(['make', 'gpufs'])
        make_path = '/home/hyf/Ganymede/Comparision/gpufs/libgpufs'
        call(['make', '-C', make_path])
        call(['make', '-C', make_path, 'install'])
        extra_objects.append(find_static_lib(
            'gpufs', [os.path.join(gpufs_install_dir, 'lib')]))

    os.chdir(this_dir)

class UninstallCommand(install):
    """Custom command to uninstall the package."""
    def run(self):
        call(['pip', 'uninstall', '-y', 'Geminifs'])
        install_dir = os.path.join(sys.prefix, 'lib', 'python3.x', 'site-packages', 'Geminifs')  # 替换 python 版本
        if os.path.exists(install_dir):
            shutil.rmtree(install_dir)
        if enable_gpufs:
            make_path = '/home/hyf/Ganymede/Comparision/gpufs/libgpufs'
            call(['make', '-C', make_path, 'clean'])

            
if sys.argv[1] in ('install', 'develop', 'bdist_wheel'):
    setup_dependencies()
    from torch.utils.cpp_extension import BuildExtension
    ext_modules.append(cpp_ext_helper('Geminifs._C', sources,
                                      extra_objects=extra_objects,
                                      libraries=libraries,
                                      define_macros=define_macros
                                      ))
    cmdclass['build_ext'] = BuildExtension

cmdclass['uninstall'] = UninstallCommand

def get_version():
    with open('interface_for_pytorch/version.txt') as f:
        version = f.read().strip()
        return version

def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]

setup(
    name='Geminifs',
    version=get_version(),
    packages=find_packages(exclude=(
        'csrc',
        'include',
    )),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    description='Geminifs interface for save and load checkpoints',
  
    install_requires=fetch_requirements('interface_for_pytorch/requirements.txt'),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
