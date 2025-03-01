# add aliengo_py/__init__.py as a package
# the corresponding compiled lib is aliengo_py/aliengo_py.cpython-38-aarch64-linux-gnu.so

from setuptools import setup, find_packages

setup(
    name="aliengo_py",
    version="0.1",
    packages=["aliengo_py"],
    package_data={"aliengo_py": ["*.so"]},
    include_package_data=True,
    install_requires=["numpy"],
    zip_safe=False,
)
