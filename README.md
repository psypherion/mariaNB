# Software Requirements Specification (SRS)
for
**Marianb: Jupyter Notebook Extension for MariaDB**
Version 1.1

Date: August 29, 2025

---

## 1. Introduction
### 1.1 Purpose
The purpose of Marianb is to enhance the workflow of data scientists, analysts, and developers who use MariaDB within Jupyter Notebooks. By providing SQL magics, DataFrame integration, plotting, connection pooling, and vector search utilities, Marianb simplifies data exploration and analysis.

### 1.2 Scope
Marianb integrates MariaDB seamlessly into Jupyter Notebooks. Users can execute SQL queries with magic commands, fetch results as DataFrames, perform vector-based semantic search, and visualize results—all without leaving the notebook environment.

### 1.3 Definitions, Acronyms, and Abbreviations
- **SQL Magics**: Jupyter magic commands for executing SQL queries.
- **DataFrame**: A Pandas data structure for tabular data.
- **Vector Search**: Approximate nearest neighbor search over MariaDB’s VECTOR(N) type.
- **Connection Pooling**: Reusing existing database connections for efficiency.

### 1.4 References
- MariaDB Connector/Python Documentation
- Jupyter Notebook Magic Commands Documentation
- Pandas Official Documentation

---

## 2. Overall Description
### 2.1 Product Perspective
Marianb builds on the existing MariaDB Connector/Python and extends Jupyter Notebook through magics and helper functions. It acts as a bridge between MariaDB and Python’s data analysis ecosystem.

### 2.2 Product Functions
- Execute SQL queries via `%mariadb` (line magic) or `%%mariadb` (cell magic).
- Fetch results directly as Pandas DataFrames.
- Plot query results inline with simple commands.
- Support for connection pooling and multiple connection profiles.
- Provide vector helpers for MariaDB’s VECTOR(N) type and ANN search.
- Enable seamless export of SQL results into Python workflows.

### 2.3 User Classes and Characteristics
- **Data Scientists**: Use Marianb for analysis, visualization, and vector similarity search.
- **Developers**: Integrate MariaDB into notebooks for prototyping and testing.
- **Educators/Students**: Learn SQL and database concepts interactively in Jupyter.

### 2.4 Operating Environment
- Jupyter Notebook / JupyterLab
- Python 3.8+
- MariaDB Server 11.0+
- Pandas, Matplotlib (optional for plotting)

### 2.5 Design and Implementation Constraints
- Must comply with Python DB API 2.0 (PEP-249).
- Must leverage MariaDB Connector/Python.
- Must remain lightweight and easy to install.

### 2.6 Assumptions and Dependencies
- Users have MariaDB installed and accessible.
- Users have Jupyter Notebook or JupyterLab installed.
- Users have Python environment configured with required dependencies.

---

## 3. System Features

### 3.1 SQL Execution Magics
- `%mariadb <query>` for single-line SQL queries.
- `%%mariadb <query>` for multi-line SQL queries.
- Output can be shown as tables or converted into Pandas DataFrames with `-df` option.

### 3.2 DataFrame Conversion
- Automatic conversion of query results into Pandas DataFrames.
- `%sql2dataframe <query>` creates a DataFrame instantly.

### 3.3 Plotting Helpers
- Inline plotting via `df.maria.plot("bar")` or `%mariadb --plot`.

### 3.4 Connection Management
- Connection pooling using Connector/Python pool.
- Saved session profiles for multiple projects (`--profile`).

### 3.5 Vector Helpers
- Create VECTOR columns in MariaDB.
- Insert embeddings from DataFrames.
- Perform ANN search with cosine or other distance metrics.

---

## 4. External Interface Requirements
### 4.1 User Interface
- Jupyter Notebook cell magics.
- Inline DataFrame and plotting outputs.

### 4.2 Software Interfaces
- MariaDB Connector/Python
- Pandas
- Matplotlib
- Jupyter Notebook / JupyterLab

### 4.3 Communications Interfaces
- TCP/IP connection to MariaDB Server.

---

## 5. Non-Functional Requirements
### 5.1 Performance Requirements
- Query execution latency should be minimal and close to native MariaDB Connector performance.
- DataFrame conversion should support datasets up to 1M rows efficiently.

### 5.2 Security Requirements
- Secure handling of credentials (do not expose passwords in plain text).
- Support for SSL/TLS connections to MariaDB.

### 5.3 Usability Requirements
- Minimal configuration.
- Intuitive magic commands.
- Documentation and example notebooks.

### 5.4 Portability Requirements
- Must run on Linux, macOS, and Windows environments supporting Jupyter.

---

## 6. Future Enhancements
- Integration with ML frameworks (scikit-learn, PyTorch) for training models directly from MariaDB datasets.
- Support for dashboards within Jupyter using interactive widgets.
- Extended visualization (heatmaps, scatter plots, etc.).

---

## 7. Appendix
- Example Magic Usage:
```python
%mariadb SELECT COUNT(*) FROM airports;

%%mariadb
SELECT name, city FROM airports WHERE country='India' LIMIT 5;

%sql2dataframe SELECT * FROM flights LIMIT 1000
```

- Example Vector Usage:
```python
from mariadb_ipython.vector import create_vector_table, insert_embeddings, search_neighbors

create_vector_table("airports", dim=384)
insert_embeddings(df, "airports", col="embedding")
search_neighbors("airports", query_vector, k=5, metric="cosine")
```

