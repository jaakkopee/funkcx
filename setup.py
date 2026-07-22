from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import pybind11


here = Path(__file__).resolve().parent


class ObjectiveCPlusPlusBuildExt(build_ext):
    def build_extensions(self):
        if ".mm" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".mm")
        super().build_extensions()


ext_modules = [
    Extension(
        name="_metal_nn",
        sources=[str(here / "metal_backend.mm")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17", "-fobjc-arc"],
        extra_link_args=["-framework", "Foundation", "-framework", "Metal"],
    )
]


setup(
    name="funkcx-metal-backend",
    version="0.1.0",
    description="Optional Metal dense-layer backend for funkcx",
    ext_modules=ext_modules,
    cmdclass={"build_ext": ObjectiveCPlusPlusBuildExt},
)