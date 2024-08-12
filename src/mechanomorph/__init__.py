"""A library for quantification of mechanical and morphological features."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mechanomorph")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"
