from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pyayay",
        sources = [
            "src/wrapper.cpp",
            "src/aychip.cpp",
        ],
        include_dirs = ["src"],
    ),
]

setup(
    name="pyayay",
    author="Ruslan Grohovecki",
    author_email="ruslan.gr@gmail.com",
    description="PyAYay is a Python extension wrapper for AY-3-8910 sound chip emulator",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    setup_requires=["setuptools_scm"],
)
