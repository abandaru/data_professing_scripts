from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize a Spark session
spark = SparkSession.builder.appName("TransposeExample").getOrCreate()

# Sample data
data = [("A","D1", "X", 10, 20),
        ("A", "D2", "Y", 20, 30),
        ("B", "D1", "X", 30, 40),
        ("B", "D2", "Y", 40, 50)]

columns = ["KeyField", "KeyField2",  "PivotField", "Value1", "value2"]

input_df = spark.createDataFrame(data, columns)
input_df.show()

# Configuration for transpose
key_fields = ["KeyField"]
data_field = [ "KeyField2", "PivotField"]
variables = ["Value1", "value2"]

transpose_to_rows = False  # Set to True for transposing to rows, False for transposing to columns

# Transpose using pivot
if transpose_to_rows:
    transposed_df = input_df.groupBy(*key_fields).pivot(data_field).agg(F.first("Value1"))
else:
    transposed_df = input_df.groupBy(data_field).pivot(*key_fields).agg(F.first("Value1"))

# Show the transposed DataFrame
transposed_df.show(truncate=False)


+--------+---------+----------+------+------+
|KeyField|KeyField2|PivotField|Value1|value2|
+--------+---------+----------+------+------+
|       A|       D1|         X|    10|    20|
|       A|       D2|         Y|    20|    30|
|       B|       D1|         X|    30|    40|
|       B|       D2|         Y|    40|    50|
+--------+---------+----------+------+------+

+---------+----------+---+---+
|KeyField2|PivotField|A  |B  |
+---------+----------+---+---+
|D2       |Y         |20 |40 |
|D1       |X         |10 |30 |
+---------+----------+---+---+

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import array, col, explode, lit, struct


# Initialize a Spark session
spark = SparkSession.builder.appName("TransposeExample").getOrCreate()

# Sample data
data = [("A","D1", "X", 10, 20),
        ("A", "D2", "Y", 20, 30),
        ("B", "D1", "X", 30, 40),
        ("B", "D2", "Y", 40, 50)]

columns = ["KeyField", "KeyField2",  "PivotField", "Value1", "value2"]

input_df = spark.createDataFrame(data, columns)
input_df.show()

# Configuration for transpose
key_fields = ["KeyField"]
data_field = ["PivotField"]
variables = [ "Value1", "value2"]


transpose_to_rows = False  # Set to True for transposing to rows, False for transposing to columns

# Transpose using pivot
if transpose_to_rows:
    transposed_df = input_df.groupBy(*key_fields).pivot(data_field).agg(*[F.first(c) for c in variables])
else:
    transposed_df = input_df.groupBy(*data_field).pivot(*key_fields).agg(*[F.first(c) for c in variables])

# Show the transposed DataFrame
transposed_df.show(truncate=False)

+--------+---------+----------+------+------+
|KeyField|KeyField2|PivotField|Value1|value2|
+--------+---------+----------+------+------+
|       A|       D1|         X|    10|    20|
|       A|       D2|         Y|    20|    30|
|       B|       D1|         X|    30|    40|
|       B|       D2|         Y|    40|    50|
+--------+---------+----------+------+------+

+----------+---------------+---------------+---------------+---------------+
|PivotField|A_first(Value1)|A_first(value2)|B_first(Value1)|B_first(value2)|
+----------+---------------+---------------+---------------+---------------+
|Y         |20             |30             |40             |50             |
|X         |10             |20             |30             |40             |
+----------+---------------+---------------+---------------+---------------+


from pyspark.sql.functions import array, col, explode, lit, struct


# Create a DataFrame
data = [
    (1, 1, 2, 3, 8, 4, 5),
    (2, 4, 3, 8, 7, 9, 8),
    (3, 6, 1, 9, 2, 3, 6),
    (4, 7, 8, 6, 9, 4, 5),
    (5, 9, 2, 7, 8, 7, 3),
    (6, 1, 1, 4, 2, 8, 4)
]

columns = ["uid", "col1", "col2", "col3", "col4", "col5", "col6"]
df = spark.createDataFrame(data, columns)

df.show(10, False)

# Create the transpose user-defined function.
def transpose_udf(trans_df, trans_by):
    cols = [c for c in trans_df.columns if c not in trans_by]
    types = [trans_df.schema[c].dataType for c in cols]
    assert len(set(types)) == 1, "All columns to transpose must have the same data type"

    kvs = F.explode(F.array([F.struct(F.lit(c).alias("column_name"), F.col(c).alias("column_value")) for c in cols]))

    by_exprs = [F.col(trans_by_col) for trans_by_col in trans_by]

    return trans_df.select(by_exprs + [kvs.alias("_kvs")]) \
        .select(by_exprs + [F.col("_kvs.column_name"), F.col("_kvs.column_value")])

# Apply the transpose user-defined function
transposed_df = transpose_udf(df, ["uid"])
transposed_df.show(50, False)

+---+----+----+----+----+----+----+
|uid|col1|col2|col3|col4|col5|col6|
+---+----+----+----+----+----+----+
|1  |1   |2   |3   |8   |4   |5   |
|2  |4   |3   |8   |7   |9   |8   |
|3  |6   |1   |9   |2   |3   |6   |
|4  |7   |8   |6   |9   |4   |5   |
|5  |9   |2   |7   |8   |7   |3   |
|6  |1   |1   |4   |2   |8   |4   |
+---+----+----+----+----+----+----+

+---+-----------+------------+
|uid|column_name|column_value|
+---+-----------+------------+
|1  |col1       |1           |
|1  |col2       |2           |
|1  |col3       |3           |
|1  |col4       |8           |
|1  |col5       |4           |
|1  |col6       |5           |
|2  |col1       |4           |
|2  |col2       |3           |
|2  |col3       |8           |
|2  |col4       |7           |
|2  |col5       |9           |
|2  |col6       |8           |
|3  |col1       |6           |
|3  |col2       |1           |
|3  |col3       |9           |
|3  |col4       |2           |
|3  |col5       |3           |
|3  |col6       |6           |
|4  |col1       |7           |
|4  |col2       |8           |
|4  |col3       |6           |
|4  |col4       |9           |
|4  |col5       |4           |
|4  |col6       |5           |
|5  |col1       |9           |
|5  |col2       |2           |
|5  |col3       |7           |
|5  |col4       |8           |
|5  |col5       |7           |
|5  |col6       |3           |
|6  |col1       |1           |
|6  |col2       |1           |
|6  |col3       |4           |
|6  |col4       |2           |
|6  |col5       |8           |
|6  |col6       |4           |
+---+-----------+------------+

from pyspark.sql import SparkSession
import yaml

def generate_spark_sql(step_config):
    # Extract step configuration
    step_number = step_config['step']
    help_text = step_config.get('help', f"Step {step_number}")

    # Extract SQL configuration parameters
    sql = step_config['sql']
    params = step_config.get('params', {})

    # Replace parameter values in SQL
    for param_name, param_value in params.items():
        sql = sql.replace(f":{param_name}", "'" + str(param_value) + "'")

    # Generate Spark SQL query
    spark_sql_query = f"""
    -- Step {step_number}: {help_text}
    {sql}
    """
    
    return spark_sql_query

if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder.appName("YAML to Spark SQL").getOrCreate()

    # Path to the YAML configuration file
    yaml_file_path = "transformations.yaml"

    # Read YAML file
    with open(yaml_file_path, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    # Iterate through each step and generate Spark SQL query
    for step_config in yaml_data:
        spark_sql_query = generate_spark_sql(step_config)
        print(spark_sql_query)

    # Stop Spark session
    spark.stop()


# unpivot data in pyspark
from pyspark.sql.functions import array, col, explode, lit, struct
from pyspark.sql import DataFrame
from typing import Iterable

def melt_df(
        df: DataFrame,
        id_vars: Iterable[str], value_vars: Iterable[str],
        var_name: str="variable", value_name: str="value"):
    """Convert :class:`DataFrame` from wide to long format."""

    # Create array
    _vars_and_vals = array(*(
        struct(lit(c).alias(var_name), col(c).alias(value_name))
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", explode(_vars_and_vals))

    cols = id_vars + [
            col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)



# column headers
table_columns = ["002_2022","003_2022","004_2022","005_2022","006_2022","007_2022","008_2022","009_2022","010_2022","011_2022","012_2022","001_2023","002_2023"]



df_rel = melt_df(sparkDF, ['PLANT','FRANCHISE','Product'], table_columns, 'Month', 'Forecast')



df_rel.show()