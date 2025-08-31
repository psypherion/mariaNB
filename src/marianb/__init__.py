"""
Marianb: Jupyter Notebook Extension for MariaDB - HACKATHON VERSION
"""

__version__ = "0.1.0"
__author__ = "@psypherion"

from .magic import MariaDBMagics
from .connection import ConnectionManager
from .dataframe import DataFrameConverter, DataFrameAccessor
from .vector import VectorUtils

def load_ipython_extension(ipython):
    """Load the Marianb extension in Jupyter"""
    print("🚀 Loading Marianb - MariaDB Hackathon Edition v0.1.0")
    
    # Register the magic commands
    magics = MariaDBMagics(shell=ipython)
    
    # Register line magic as 'mariadb'
    ipython.register_magic_function(magics.mariadb_line, 'line', 'mariadb')
    
    # Register cell magic as 'mariadb'
    ipython.register_magic_function(magics.mariadb_cell, 'cell', 'mariadb')
    
    print("✅ Marianb loaded successfully!")
    print("   📝 Use %mariadb for single-line queries")
    print("   📄 Use %%mariadb for multi-line queries") 
    print("   🎯 Example: %mariadb SELECT VERSION()")
    print("   🏆 Ready for MariaDB Hackathon!")

def unload_ipython_extension(ipython):
    """Unload the extension"""
    print("👋 Marianb hackathon edition unloaded")

__all__ = [
    'MariaDBMagics',
    'ConnectionManager', 
    'DataFrameConverter',
    'VectorUtils',
    'load_ipython_extension',
    'unload_ipython_extension'
]
