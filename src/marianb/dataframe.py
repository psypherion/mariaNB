"""
DataFrame Integration Module for Marianb
Handles conversion between MariaDB results and pandas DataFrames with optimizations
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
from datetime import datetime, date, time
import json
import logging

logger = logging.getLogger(__name__)

class DataFrameConverter:
    """Handles conversion between MariaDB and pandas DataFrames with optimizations"""
    
    def __init__(self):
        self.type_mapping = self._initialize_type_mapping()
        self.chunk_size = 10000  # Default chunk size for large operations
    
    def _initialize_type_mapping(self) -> Dict[str, str]:
        """Initialize MariaDB to pandas type mapping for optimal performance"""
        return {
            # Integer types
            'TINYINT': 'Int8',
            'SMALLINT': 'Int16', 
            'MEDIUMINT': 'Int32',
            'INT': 'Int32',
            'INTEGER': 'Int32',
            'BIGINT': 'Int64',
            
            # Floating point types
            'FLOAT': 'float32',
            'DOUBLE': 'float64',
            'DECIMAL': 'float64',
            'NUMERIC': 'float64',
            
            # String types
            'CHAR': 'string',
            'VARCHAR': 'string',
            'TEXT': 'string',
            'TINYTEXT': 'string',
            'MEDIUMTEXT': 'string',
            'LONGTEXT': 'string',
            
            # Date/Time types
            'DATE': 'datetime64[ns]',
            'TIME': 'timedelta64[ns]',
            'DATETIME': 'datetime64[ns]',
            'TIMESTAMP': 'datetime64[ns]',
            'YEAR': 'Int16',
            
            # Binary types
            'BINARY': 'object',
            'VARBINARY': 'object',
            'BLOB': 'object',
            'TINYBLOB': 'object',
            'MEDIUMBLOB': 'object',
            'LONGBLOB': 'object',
            
            # JSON and Vector types
            'JSON': 'object',
            'VECTOR': 'object',
            
            # Boolean
            'BOOLEAN': 'boolean',
            'BOOL': 'boolean',
        }
    
    def sql_to_dataframe(self, cursor, optimize_types: bool = True, 
                        parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert SQL cursor results to optimized pandas DataFrame
        
        Args:
            cursor: Database cursor with executed query
            optimize_types: Whether to optimize data types for memory efficiency
            parse_dates: List of columns to parse as dates
            
        Returns:
            Optimized pandas DataFrame
        """
        try:
            # Get column information
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            if not columns:
                return pd.DataFrame()
            
            # Fetch all results
            rows = cursor.fetchall()
            
            if not rows:
                return pd.DataFrame(columns=columns)
            
            # Create initial DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            if optimize_types:
                df = self._optimize_dtypes(df, cursor.description)
            
            if parse_dates:
                df = self._parse_dates(df, parse_dates)
            
            # Handle vector columns
            df = self._process_vector_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting SQL results to DataFrame: {e}")
            raise
    
    def _optimize_dtypes(self, df: pd.DataFrame, column_info: List[Tuple]) -> pd.DataFrame:
        """Optimize DataFrame data types based on MariaDB column information"""
        try:
            for i, (col_name, col_type, _, _, _, _, _) in enumerate(column_info):
                if col_name not in df.columns:
                    continue
                
                # Get the MariaDB type name
                type_name = self._extract_type_name(str(col_type))
                
                # Skip optimization for columns with all null values
                if df[col_name].isna().all():
                    continue
                
                # Apply type optimization
                if type_name in self.type_mapping:
                    target_type = self.type_mapping[type_name]
                    try:
                        if target_type.startswith('Int'):
                            # Use nullable integer types
                            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype(target_type)
                        elif target_type in ['float32', 'float64']:
                            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype(target_type)
                        elif target_type == 'string':
                            df[col_name] = df[col_name].astype('string')
                        elif target_type == 'boolean':
                            df[col_name] = df[col_name].astype('boolean')
                        elif target_type.startswith('datetime64'):
                            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Could not optimize type for column {col_name}: {e}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Type optimization failed: {e}")
            return df
    
    def _extract_type_name(self, col_type_str: str) -> str:
        """Extract base type name from MariaDB column type string"""
        # Handle cases like "VARCHAR(255)" -> "VARCHAR"
        type_name = str(col_type_str).upper().split('(')[0]
        return type_name.strip()
    
    def _parse_dates(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Parse specified columns as dates with error handling"""
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not parse dates for column {col}: {e}")
        return df
    
    def _process_vector_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process VECTOR columns for proper handling in pandas"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if this might be a vector column
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample_val, (bytes, str)) and self._looks_like_vector(sample_val):
                    try:
                        df[col] = df[col].apply(self._parse_vector_value)
                    except Exception as e:
                        logger.warning(f"Could not process vector column {col}: {e}")
        return df
    
    def _looks_like_vector(self, value) -> bool:
        """Check if a value looks like a vector representation"""
        if isinstance(value, bytes):
            return True  # Binary vector data
        if isinstance(value, str):
            # Check for JSON array format
            try:
                parsed = json.loads(value)
                return isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed)
            except:
                return False
        return False
    
    def _parse_vector_value(self, value):
        """Parse vector value to numpy array"""
        if pd.isna(value):
            return None
        
        if isinstance(value, bytes):
            # Handle binary vector data
            try:
                # Assume float32 vectors (common format)
                return np.frombuffer(value, dtype=np.float32)
            except:
                return value
        
        if isinstance(value, str):
            try:
                # Parse JSON array
                parsed = json.loads(value)
                return np.array(parsed, dtype=np.float32)
            except:
                return value
        
        return value
    
    def dataframe_to_sql(self, df: pd.DataFrame, table_name: str, connection,
                        if_exists: str = 'append', index: bool = False,
                        method: Optional[str] = 'multi', chunksize: Optional[int] = None) -> int:
        """
        Insert DataFrame into MariaDB table with optimizations
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            connection: Database connection
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            index: Whether to write DataFrame index
            method: Insert method ('multi' for batch inserts)
            chunksize: Number of rows per chunk
            
        Returns:
            Number of rows inserted
        """
        try:
            # Use chunking for large DataFrames
            chunk_size = chunksize or self.chunk_size
            
            if len(df) > chunk_size:
                return self._chunked_insert(df, table_name, connection, if_exists, 
                                          index, method, chunk_size)
            else:
                # Direct insert for smaller DataFrames
                return df.to_sql(name=table_name, con=connection, if_exists=if_exists,
                               index=index, method=method)
                
        except Exception as e:
            logger.error(f"Error inserting DataFrame to SQL: {e}")
            raise
    
    def _chunked_insert(self, df: pd.DataFrame, table_name: str, connection,
                       if_exists: str, index: bool, method: str, chunk_size: int) -> int:
        """Insert DataFrame in chunks for better performance"""
        total_inserted = 0
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            
            # For first chunk, use the specified if_exists behavior
            # For subsequent chunks, always append
            chunk_if_exists = if_exists if i == 0 else 'append'
            
            try:
                rows_inserted = chunk.to_sql(name=table_name, con=connection,
                                           if_exists=chunk_if_exists, index=index,
                                           method=method)
                total_inserted += rows_inserted or len(chunk)
                
                logger.info(f"Inserted chunk {i//chunk_size + 1}: {len(chunk)} rows")
                
            except Exception as e:
                logger.error(f"Error inserting chunk {i//chunk_size + 1}: {e}")
                raise
        
        return total_inserted
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        initial_memory = df.memory_usage(deep=True).sum()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype('category')
                except:
                    pass
        
        final_memory = df.memory_usage(deep=True).sum()
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        logger.info(f"Memory optimization: {reduction:.1f}% reduction "
                   f"({initial_memory/1024**2:.1f}MB -> {final_memory/1024**2:.1f}MB)")
        
        return df
    
    def create_dataframe_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive DataFrame summary for analysis"""
        summary = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'null_counts': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head().to_dict()
            }
        
        return summary


class DataFrameAccessor:
    """Custom pandas accessor for MariaDB-specific operations"""
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def to_mariadb(self, table_name: str, connection, **kwargs):
        """Convenience method to insert DataFrame to MariaDB"""
        converter = DataFrameConverter()
        return converter.dataframe_to_sql(self._obj, table_name, connection, **kwargs)
    
    def vector_similarity(self, vector_column: str, query_vector: Union[List, np.ndarray],
                         metric: str = 'cosine') -> pd.Series:
        """Calculate vector similarity for a DataFrame column"""
        if vector_column not in self._obj.columns:
            raise ValueError(f"Column '{vector_column}' not found")
        
        query_vec = np.array(query_vector)
        similarities = []
        
        for vec in self._obj[vector_column]:
            if vec is None or not isinstance(vec, np.ndarray):
                similarities.append(np.nan)
                continue
            
            if metric == 'cosine':
                sim = np.dot(vec, query_vec) / (np.linalg.norm(vec) * np.linalg.norm(query_vec))
            elif metric == 'euclidean':
                sim = -np.linalg.norm(vec - query_vec)  # Negative for descending order
            else:
                sim = np.nan
            
            similarities.append(sim)
        
        return pd.Series(similarities, index=self._obj.index, name=f'{vector_column}_similarity')


# Register the custom accessor
pd.api.extensions.register_dataframe_accessor("maria")(DataFrameAccessor)


# Utility functions
def quick_query_to_df(query: str, connection, **kwargs) -> pd.DataFrame:
    """Quick utility function to execute query and return optimized DataFrame"""
    converter = DataFrameConverter()
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        return converter.sql_to_dataframe(cursor, **kwargs)
    finally:
        cursor.close()


def df_info_summary(df: pd.DataFrame, show_vector_info: bool = True) -> str:
    """Generate a comprehensive info summary for DataFrames with vector data"""
    converter = DataFrameConverter()
    summary = converter.create_dataframe_summary(df)
    
    info_lines = [
        f"DataFrame Info:",
        f"  Shape: {summary['shape']}",
        f"  Memory: {summary['memory_usage_mb']:.2f} MB",
        f"  Null values: {sum(summary['null_counts'].values())} total",
        f"",
        f"Column Types:",
    ]
    
    for col, dtype in summary['dtypes'].items():
        null_count = summary['null_counts'][col]
        info_lines.append(f"  {col}: {dtype} ({null_count} nulls)")
    
    if show_vector_info:
        # Check for potential vector columns
        vector_cols = [col for col in df.columns 
                      if df[col].dtype == 'object' and 
                      any(isinstance(val, np.ndarray) for val in df[col].dropna().iloc[:5])]
        
        if vector_cols:
            info_lines.extend([
                f"",
                f"Vector Columns: {', '.join(vector_cols)}"
            ])
    
    return "\n".join(info_lines)
