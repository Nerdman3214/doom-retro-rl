from setuptools import setup, Extension

module = Extension(
    "frame_processor",
    sources=["frame_processor.c"],
    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
)

setup(
    name="frame_processor",
    version="1.0",
    description="Fast C frame processing for DOOM RL pipeline",
    ext_modules=[module],
)
