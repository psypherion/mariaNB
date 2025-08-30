from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
import pandas as pd
import mariadb
from .connection import ConnectionManager
from .dataframe import DataFrameConverter
import sqlparse

@magics_class
class MariaDBMagics(Magics):
    
    def __init__(self, shell=None):
        super(MariaDBMagics, self).__init__(shell)
        self.connection_manager = ConnectionManager()
        self.df_converter = DataFrameConverter()
        
    @magic_arguments()
    @argument('-df', '--dataframe', type=str, 
              help='Store result in variable as DataFrame')
    @argument('-p', '--profile', type=str, default='default',
              help='Connection profile to use')
    @argument('--plot', choices=['bar', 'line', 'scatter', 'hist'],
              help='Plot the results')
    @line_magic
    def mariadb_line(self, line):
        """Execute single-line MariaDB query"""
        try:
            args = parse_argstring(self.mariadb_line, line)
            
            # Extract SQL from line (remove the arguments)
            sql_parts = []
            line_parts = line.split()
            skip_next = False
            
            for i, part in enumerate(line_parts):
                if skip_next:
                    skip_next = False
                    continue
                if part.startswith('-'):
                    if '=' not in part and i + 1 < len(line_parts) and not line_parts[i + 1].startswith('-'):
                        skip_next = True
                    continue
                sql_parts.append(part)
            
            sql = ' '.join(sql_parts)
            return self._execute_query(sql, args)
        except:
            # Fallback: treat entire line as SQL
            args = type('Args', (), {'dataframe': None, 'profile': 'default', 'plot': None})()
            return self._execute_query(line.strip(), args)
    
    @magic_arguments()
    @argument('-df', '--dataframe', type=str,
              help='Store result in variable as DataFrame')
    @argument('-p', '--profile', type=str, default='default',
              help='Connection profile to use')
    @argument('--plot', choices=['bar', 'line', 'scatter', 'hist'],
              help='Plot the results')
    @cell_magic
    def mariadb_cell(self, line, cell):
        """Execute multi-line MariaDB query"""
        try:
            args = parse_argstring(self.mariadb_cell, line)
        except:
            args = type('Args', (), {'dataframe': None, 'profile': 'default', 'plot': None})()
        
        return self._execute_query(cell, args)
    
    def _execute_query(self, sql, args):
        """Core query execution logic"""
        try:
            # Clean and validate SQL
            sql = sql.strip()
            if not sql:
                return "❌ Empty SQL query"
            
            # Get connection
            conn = self.connection_manager.get_connection(args.profile)
            
            # Execute query
            if sql.upper().startswith(('SELECT', 'WITH', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
                try:
                    # Use SQLAlchemy engine for pandas compatibility
                    from sqlalchemy import create_engine
                    
                    # Get connection details
                    conn_details = self.connection_manager.profiles[args.profile]
                    engine = create_engine(f"mysql+pymysql://{conn_details['user']}:{conn_details['password']}@{conn_details['host']}:{conn_details['port']}/{conn_details['database']}")
                    
                    df = pd.read_sql(sql, engine)
                except ImportError:
                    # Fallback to direct connection (with warning)
                    df = pd.read_sql(sql, conn)

                
                # Handle DataFrame storage
                if hasattr(args, 'dataframe') and args.dataframe:
                    self.shell.user_ns[args.dataframe] = df
                    print(f"✅ Result stored in variable '{args.dataframe}'")
                
                # Handle plotting
                if hasattr(args, 'plot') and args.plot:
                    self._plot_dataframe(df, args.plot)
                
                return df
            else:
                # Non-SELECT queries
                cursor = conn.cursor()
                cursor.execute(sql)
                rows_affected = cursor.rowcount
                conn.commit()
                cursor.close()
                return f"✅ Query executed successfully. {rows_affected} rows affected."
                
        except Exception as e:
            return f"❌ Error executing query: {str(e)}"
    
    def _plot_dataframe(self, df, plot_type):
        """Generate inline plots for DataFrame results"""
        try:
            import matplotlib.pyplot as plt
            
            if plot_type == 'bar':
                df.plot(kind='bar', figsize=(10, 6))
            elif plot_type == 'line':
                df.plot(kind='line', figsize=(10, 6))
            elif plot_type == 'scatter' and len(df.columns) >= 2:
                df.plot(kind='scatter', x=df.columns[0], y=df.columns[1], figsize=(10, 6))
            elif plot_type == 'hist':
                df.hist(figsize=(12, 8))
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("⚠️  Matplotlib not available for plotting")
        except Exception as e:
            print(f"⚠️  Plotting error: {str(e)}")
