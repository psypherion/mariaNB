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
    def mariadb(self, line):
        """Execute single-line MariaDB query"""
        args = parse_argstring(self.mariadb, line)
        
        # Extract SQL from remaining line
        sql = line.split(None, len(line.split()) - len([x for x in line.split() if x.startswith('-')]))
        sql = ' '.join(sql[len([x for x in line.split() if x.startswith('-')]):])
        
        return self._execute_query(sql, args)
    
    @magic_arguments()
    @argument('-df', '--dataframe', type=str,
              help='Store result in variable as DataFrame')
    @argument('-p', '--profile', type=str, default='default',
              help='Connection profile to use')
    @argument('--plot', choices=['bar', 'line', 'scatter', 'hist'],
              help='Plot the results')
    @cell_magic
    def mariadb(self, line, cell):
        """Execute multi-line MariaDB query"""
        args = parse_argstring(self.mariadb, line)
        return self._execute_query(cell, args)
    
    def _execute_query(self, sql, args):
        """Core query execution logic"""
        try:
            # Clean and validate SQL
            sql = sql.strip()
            if not sql:
                raise ValueError("Empty SQL query")
            
            # Get connection
            conn = self.connection_manager.get_connection(args.profile)
            
            # Execute query
            if sql.upper().startswith('SELECT') or sql.upper().startswith('WITH'):
                df = pd.read_sql(sql, conn)
                
                # Handle DataFrame storage
                if args.dataframe:
                    self.shell.user_ns[args.dataframe] = df
                    print(f"Result stored in variable '{args.dataframe}'")
                
                # Handle plotting
                if args.plot:
                    self._plot_dataframe(df, args.plot)
                
                return df
            else:
                # Non-SELECT queries
                cursor = conn.cursor()
                cursor.execute(sql)
                rows_affected = cursor.rowcount
                conn.commit()
                cursor.close()
                return f"Query executed successfully. {rows_affected} rows affected."
                
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def _plot_dataframe(self, df, plot_type):
        """Generate inline plots for DataFrame results"""
        try:
            import matplotlib.pyplot as plt
            
            if plot_type == 'bar':
                df.plot(kind='bar')
            elif plot_type == 'line':
                df.plot(kind='line')
            elif plot_type == 'scatter' and len(df.columns) >= 2:
                df.plot(kind='scatter', x=df.columns[0], y=df.columns[1])
            elif plot_type == 'hist':
                df.hist()
            
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Plotting error: {str(e)}")

# IPython extension loading
def load_ipython_extension(ipython):
    """Called when extension is loaded"""
    ipython.register_magic_function(MariaDBMagics)
