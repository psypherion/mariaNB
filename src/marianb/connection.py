import mariadb
from typing import Dict, Optional
import os
import json
from pathlib import Path

class ConnectionManager:
    """Manages MariaDB connections with pooling and profiles"""
    
    def __init__(self):
        self.pools: Dict[str, mariadb.ConnectionPool] = {}
        self.config_file = Path.home() / '.marianb' / 'connections.json'
        self.load_profiles()
    
    def load_profiles(self):
        """Load connection profiles from config file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.profiles = json.load(f)
        else:
            # Default profile
            self.profiles = {
                'default': {
                    'host': os.getenv('MARIADB_HOST', 'localhost'),
                    'port': int(os.getenv('MARIADB_PORT', 3306)),
                    'user': os.getenv('MARIADB_USER', 'root'),
                    'password': os.getenv('MARIADB_PASSWORD', ''),
                    'database': os.getenv('MARIADB_DATABASE', 'test'),
                    'pool_name': 'marianb_default',
                    'pool_size': 5
                }
            }
    
    def get_connection(self, profile_name: str = 'default'):
        """Get connection from pool for specified profile"""
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        profile = self.profiles[profile_name]
        pool_name = profile.get('pool_name', f'marianb_{profile_name}')
        
        # Create pool if it doesn't exist
        if pool_name not in self.pools:
            try:
                self.pools[pool_name] = mariadb.ConnectionPool(
                    user=profile['user'],
                    password=profile['password'],
                    host=profile['host'],
                    port=profile['port'],
                    database=profile['database'],
                    pool_name=pool_name,
                    pool_size=profile.get('pool_size', 5)
                )
            except mariadb.Error as e:
                raise ConnectionError(f"Failed to create connection pool: {e}")
        
        try:
            return self.pools[pool_name].get_connection()
        except mariadb.Error as e:
            raise ConnectionError(f"Failed to get connection: {e}")
    
    def add_profile(self, name: str, **kwargs):
        """Add new connection profile"""
        required_fields = ['host', 'user', 'password', 'database']
        for field in required_fields:
            if field not in kwargs:
                raise ValueError(f"Missing required field: {field}")
        
        self.profiles[name] = {
            'host': kwargs['host'],
            'port': kwargs.get('port', 3306),
            'user': kwargs['user'],
            'password': kwargs['password'],
            'database': kwargs['database'],
            'pool_name': f'marianb_{name}',
            'pool_size': kwargs.get('pool_size', 5)
        }
        
        self.save_profiles()
    
    def save_profiles(self):
        """Save profiles to config file"""
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.profiles, f, indent=2)
