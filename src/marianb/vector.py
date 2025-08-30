import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict
import mariadb

class VectorUtils:
    """Utilities for MariaDB vector operations"""
    
    def __init__(self, connection_manager):
        self.conn_manager = connection_manager
    
    def create_vector_table(self, table_name: str, vector_dim: int, 
                          profile: str = 'default', additional_columns: Optional[Dict] = None):
        """Create table with VECTOR column"""
        conn = self.conn_manager.get_connection(profile)
        cursor = conn.cursor()
        
        # Base columns
        columns = [
            "id INT AUTO_INCREMENT PRIMARY KEY",
            f"embedding VECTOR({vector_dim})"
        ]
        
        # Add additional columns if specified
        if additional_columns:
            for col_name, col_type in additional_columns.items():
                columns.append(f"{col_name} {col_type}")
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns)}
        )
        """
        
        cursor.execute(create_sql)
        
        # Create vector index for ANN search
        index_sql = f"""
        CREATE INDEX idx_{table_name}_embedding 
        ON {table_name}(embedding) 
        USING HNSW
        """
        
        try:
            cursor.execute(index_sql)
        except mariadb.Error as e:
            if "already exists" not in str(e):
                raise
        
        conn.commit()
        cursor.close()
        print(f"Vector table '{table_name}' created successfully")
    
    def insert_embeddings(self, df: pd.DataFrame, table_name: str, 
                         embedding_column: str, profile: str = 'default', 
                         additional_columns: Optional[List[str]] = None):
        """Insert DataFrame with embeddings into vector table"""
        conn = self.conn_manager.get_connection(profile)
        cursor = conn.cursor()
        
        # Prepare columns for insertion
        columns = ["embedding"]
        if additional_columns:
            columns.extend(additional_columns)
        
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)
        
        insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        
        # Prepare data for insertion
        data_to_insert = []
        for _, row in df.iterrows():
            row_data = [row[embedding_column]]  # Vector embedding
            if additional_columns:
                row_data.extend([row[col] for col in additional_columns])
            data_to_insert.append(tuple(row_data))
        
        cursor.executemany(insert_sql, data_to_insert)
        conn.commit()
        cursor.close()
        print(f"Inserted {len(data_to_insert)} embeddings into '{table_name}'")
    
    def search_neighbors(self, table_name: str, query_vector: Union[List, np.ndarray], 
                        k: int = 5, metric: str = 'cosine', profile: str = 'default',
                        additional_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform approximate nearest neighbor search"""
        conn = self.conn_manager.get_connection(profile)
        
        # Convert query vector to proper format
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        # Build query
        select_columns = ["id", "embedding"]
        if additional_columns:
            select_columns.extend(additional_columns)
        
        columns_str = ", ".join(select_columns)
        
        # MariaDB vector search with distance function
        if metric == 'cosine':
            distance_func = "VEC_DISTANCE_COSINE"
        elif metric == 'euclidean':
            distance_func = "VEC_DISTANCE_EUCLIDEAN"
        else:
            distance_func = "VEC_DISTANCE_COSINE"  # Default
        
        search_sql = f"""
        SELECT {columns_str}, 
               {distance_func}(embedding, ?) as distance
        FROM {table_name}
        ORDER BY distance ASC
        LIMIT ?
        """
        
        # Execute search
        df = pd.read_sql(search_sql, conn, params=[query_vector, k])
        return df
