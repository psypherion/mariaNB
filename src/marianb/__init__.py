"""
Marianb: Jupyter Notebook Extension for MariaDB
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .magic import MariaDBMagics
from .connection import ConnectionManager
from .dataframe import DataFrameConverter, DataFrameAccessor
from .vector import VectorUtils

# Make the extension loadable
def load_ipython_extension(ipython):
    """Load the Marianb extension in Jupyter"""
    print("🚀 Loading Marianb - MariaDB Jupyter Extension v0.1.0")
    
    # Register the magic commands
    magics = MariaDBMagics(shell=ipython)
    ipython.register_magic_function(magics.mariadb, 'line_cell')
    
    print("✅ Marianb loaded successfully!")
    print("   Use %mariadb for single-line queries")
    print("   Use %%mariadb for multi-line queries") 
    print("   Example: %mariadb SELECT VERSION()")

def unload_ipython_extension(ipython):
    """Unload the extension"""
    print("👋 Marianb extension unloaded")

# Expose main classes for direct import
__all__ = [
    'MariaDBMagics',
    'ConnectionManager', 
    'DataFrameConverter',
    'VectorUtils',
    'load_ipython_extension',
    'unload_ipython_extension'
]
