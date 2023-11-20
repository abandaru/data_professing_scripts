from typing import List, Optional, Union
from datetime import datetime
import re
from pyspark.sql import types as T
import math
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.utils import AnalysisException
import pyspark.sql.functions as F 
from pyspark.sql.functions import desc, row_number
from pyspark.sql.types import IntegerType, ArrayType, MapType, StructField, StructType 
from time import time

from strplus import Str

def get_log4j_logger(spark: SparkSession):
    """
    Gets a logger needed for logging useful information
    :param spark:  A Spark Session
    :return: A log4j logger for Spark logging
    """
    log4j_logger = spark.sparkContext._jvm.org.apache.log4j
    return log4j_logger.LogManager.getLogger(__name__)

def complex_fields(schema: T.StructType):
    """Returns a dictionary of complex field names and their data types from the input DataFrame's schema.

    :param df: The input PySpark DataFrame.
    :type df: DataFrame
    :return: A dictionary with complex field names as keys and their respective data types as values.
    :rtype: Dict[str, object]
    """
    return {
        field.name: field.dataType
        for field in schema.fields
        if isinstance(field.dataType, (T.ArrayType, T.StructType, T.MapType))
    }

def flatten_struct(df: DataFrame, col_name: str, separator: str = ":") -> DataFrame:
    """Flattens the specified StructType column in the input DataFrame and returns a new DataFrame with the flattened columns.

    :param df: The input PySpark DataFrame.
    :type df: DataFrame
    :param col_name: The column name of the StructType to be flattened.
    :type col_name: str
    :param separator: The separator to use in the resulting flattened column names, defaults to ':'.
    :type separator: str, optional
    :return: The DataFrame with the flattened StructType column.
    :rtype: List[Column]
    """
    struct_type = complex_fields(df.schema)[col_name]
    expanded = [
        F.col(f"`{col_name}`.`{k}`").alias(col_name + separator + k)
        for k in [n.name for n in struct_type.fields]
    ]
    return df.select("*", *expanded).drop(F.col(f"`{col_name}`"))


def flatten_map(df: DataFrame, col_name: str, separator: str = ":") -> DataFrame:
    """Flattens the specified MapType column in the input DataFrame and returns a new DataFrame with the flattened columns.

    :param df: The input PySpark DataFrame.
    :type df: DataFrame
    :param col_name: The column name of the MapType to be flattened.
    :type col_name: str
    :param separator: The separator to use in the resulting flattened column names, defaults to ":".
    :type separator: str, optional
    :return: The DataFrame with the flattened MapType column.
    :rtype: DataFrame
    """
    keys_df = df.select(F.explode_outer(F.map_keys(F.col(f"`{col_name}`")))).distinct()
    keys = [row[0] for row in keys_df.collect()]
    key_cols = [
        F.col(f"`{col_name}`").getItem(k).alias(col_name + separator + k) for k in keys
    ]
    return df.select(
        [F.col(f"`{col}`") for col in df.columns if col != col_name] + key_cols,
    )


def flatten_dataframe(
    df: DataFrame,
    separator: str = ":",
    replace_char: str = "_",
    sanitized_columns: bool = False,  # noqa: FBT001, FBT002
) -> DataFrame:
    """Flattens the complex columns in the DataFrame.

    :param df: The input PySpark DataFrame.
    :type df: DataFrame
    :param separator: The separator to use in the resulting flattened column names, defaults to ":".
    :type separator: str, optional
    :param replace_char: The character to replace special characters with in column names, defaults to "_".
    :type replace_char: str, optional
    :param sanitized_columns: Whether to sanitize column names, defaults to False.
    :type sanitized_columns: bool, optional
    :return: The DataFrame with all complex data types flattened.
    :rtype: DataFrame

    .. note:: This function assumes the input DataFrame has a consistent schema across all rows. If you have files with
        different schemas, process each separately instead.

    .. example:: Example usage:

        >>> data = [
                (
                    1,
                    ("Alice", 25),
                    {"A": 100, "B": 200},
                    ["apple", "banana"],
                    {"key": {"nested_key": 10}},
                    {"A#": 1000, "B@": 2000},
                ),
                (
                    2,
                    ("Bob", 30),
                    {"A": 150, "B": 250},
                    ["orange", "grape"],
                    {"key": {"nested_key": 20}},
                    {"A#": 1500, "B@": 2500},
                ),
            ]

        >>> df = spark.createDataFrame(data)
        >>> flattened_df = flatten_dataframe(df)
        >>> flattened_df.show()
        >>> flattened_df_with_hyphen = flatten_dataframe(df, replace_char="-")
        >>> flattened_df_with_hyphen.show()
    """

    def sanitize_column_name(name: str, rc: str = "_") -> str:
        """Sanitizes column names by replacing special characters with the specified character.

        :param name: The original column name.
        :type name: str
        :param rc: The character to replace special characters with, defaults to '_'.
        :type rc: str, optional
        :return: The sanitized column name.
        :rtype: str
        """
        return re.sub(r"[^a-zA-Z0-9_]", rc, name)

    def explode_array(df: DataFrame, col_name: str) -> DataFrame:
        """Explodes the specified ArrayType column in the input DataFrame and returns a new DataFrame with the exploded column.

        :param df: The input PySpark DataFrame.
        :type df: DataFrame
        :param col_name: The column name of the ArrayType to be exploded.
        :type col_name: str
        :return: The DataFrame with the exploded ArrayType column.
        :rtype: DataFrame
        """
        return df.select(
            "*", F.explode_outer(F.col(f"`{col_name}`")).alias(col_name)
        ).drop(
            col_name,
        )

    fields = complex_fields(df.schema)

    while len(fields) != 0:
        col_name = next(iter(fields.keys()))

        if isinstance(fields[col_name], StructType):
            df = flatten_struct(df, col_name, separator)  # noqa: PD901

        elif isinstance(fields[col_name], ArrayType):
            df = explode_array(df, col_name)  # noqa: PD901

        elif isinstance(fields[col_name], MapType):
            df = flatten_map(df, col_name, separator)  # noqa: PD901

        fields = complex_fields(df.schema)

    # Sanitize column names with the specified replace_char
    if sanitized_columns:
        sanitized_columns = [
            sanitize_column_name(col_name, replace_char) for col_name in df.columns
        ]
        df = df.toDF(*sanitized_columns)  # noqa: PD901

    return df

def add_salt_column(df: DataFrame, skew_factor: int) -> DataFrame:
    """
    Adds a salt column to a DataFrame. We will be using this salt column when we are trying to perform
    join, groupBy, etc. operations into a skewed DataFrame. The idea is to add a random column and use
    the original keys + this salted key to perform the operations, so that we can avoid data skewness and
    possibly, OOM errors.

    Args:
        df: A PySpark DataFrame
        skew_factor: The skew factor. For example, if we set this value to 3, then the salted column will
            be populated by the elements 0, 1 and 2, extracted from a uniform probability distribution.

    Returns:
        The original DataFrame with a `salt_id` column.
    """
    return df.withColumn("salt_id", (F.rand() * skew_factor).cast(IntegerType()))

def optimal_cross_join(df_bc: DataFrame, df: DataFrame) -> DataFrame:
    """
    A simple trick to solve the problem we have with CrossJoins between two dataframes,
    when the resulting partitions will be a multiplication of the initial partitions (e.g. if
    we make the cross join between `df1` and `df2` both with 100 partitions, then the resulting
    partitions will be 10_000).

    Args:
        df_bc: The DataFrame to be broadcasted
        df: The DataFrame

    Returns:
        The cross joined DataFrame
    """
    return F.broadcast(df_bc).crossJoin(df)

def df_size_in_bytes_exact(df: DataFrame):
    """
    Calculates the exact size in memory of a DataFrame by caching it and accessing the optimized plan

    Note: BE CAREFUL WITH THIS FUNCTION BECAUSE IT WILL CACHE ALL THE DATAFRAME!!! IF YOUR DATAFRAME IS
    TOO BIG USE `estimate_df_size_in_bytes`!!

    Args:
        df: A pyspark DataFrame

    Returns:
        The exact size in bytes
    """
    df.cache().count()
    size_in_bytes = df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()
    df.unpersist(blocking=True)
    return size_in_bytes

def df_size_in_bytes_approximate(df: DataFrame, sample_perc: float = 0.05):
    """
    This method takes a sample of the input DataFrame (`sample_perc`) and applies `df_size_in_bytes_exact`
    method to it. After it calculates the exact size of the sample, it extrapolates the total size.

    Args:
        df: A PySpark DataFrame
        sample_perc: The percentage of the DataFrame to sample. By default, a 5 %

    Raises:
        ValueError: If `sample_perc` is less than or equal to 0 or if it's greater than 1.

    Returns:
        The approximate size in bytes
    """
    if sample_perc <= 0 or sample_perc > 1:
        raise ValueError("`sample_perc` must be in the interval (0, 1]")

    sample_size_in_bytes = df_size_in_bytes_exact(df.sample(sample_perc))
    return sample_size_in_bytes / sample_perc

def remove_empty_partitions(df: DataFrame):
    """
    This method will remove empty partitions from a DataFrame. It is useful after a filter, for
    example, when a great number of partitions may contain zero registers.

    Note: This functionality may be useless if you are using Adaptive Query Execution from Spark 3.0

    Args:
        df: A pyspark DataFrame

    Returns:
        A DataFrame with all empty partitions removed
    """

    def _count_number_of_non_empty_partitions(iterator):
        """
        Simply returns de number of nonempty partitions in a DataFrame.

        Args:
            iterator: An iterator containing each partition

        Yields:
            The number of non-empty partitions
        """
        n = 0
        for _ in iterator:
            n += 1
            break
        yield n

    non_empty_partitions = sum(
        df.rdd.mapPartitions(_count_number_of_non_empty_partitions).collect()
    )
    return df.coalesce(non_empty_partitions)


def add_partition_id_col(df: DataFrame, partition_id_colname: str = "partition_id"):
    """
    Adds a column named `partition_id` to the input DataFrame which represents the partition id as
    output by `pyspark.sql.functions.spark_partition_id` method.

    Args:
        df: A PySpark DataFrame
        partition_id_colname: The name of the column containing the partition id

    Returns:
        The input DataFrame with an additional column (`partition_id`) which represents the partition id
    """
    return df.withColumn(partition_id_colname, F.spark_partition_id())


def get_partition_records_df(df: DataFrame) -> DataFrame:
    """
    Generates a DataFrame containing the number of elements for each partition. This method
    may be handy when trying to determine if data is skewed.

    Args:
        df: A PySpark DataFrame

    Returns:
        A DataFrame containing `partition_id` and `count` columns
    """
    return add_partition_id_col(df).groupBy("partition_id").count()


def get_partition_records_distribution(
    df: DataFrame, probabilities: List[float], relative_error: float = 0.0
) -> List[float]:
    """
    Generates a DataFrame containing

    Args:
        df: A PySpark DataFrame
        probabilities: The list of probabilities to be shown in the output DataFrame
        relative_error: The relative target precision. For more information, check PySpark's
            documentation about `approxQuantile` (https://spark.apache.org/docs/latest/api/python/
            reference/pyspark.sql/api/pyspark.sql.DataFrame.approxQuantile.html). Defaults to 0.
            to obtain exact quantiles, but if the operation is too expensive you can increase this
            value (although the quantile precision will diminish)

    Returns:
        A list containing the value for each probability.
    """
    partition_records_df = get_partition_records_df(df)
    return partition_records_df.approxQuantile(
        col="count", probabilities=probabilities, relativeError=relative_error
    )


def get_keys_records_df(df: DataFrame, keys: Union[str, List[str]]) -> DataFrame:
    """
    Creates a DataFrame containing the number of records per each key.

    Args:
        df: A PySpark DataFrame
        keys: A col or list of cols from the DataFrame

    Returns:
        The keys records DataFrame
    """
    return df.groupBy(keys).count()


def get_keys_records_distribution(
    df: DataFrame,
    keys: Union[str, List[str]],
    probabilities: List[float],
    relative_error: float = 0.0,
):
    """
    Calculates the distribution of the number of records over the DataFrame keys.

    Args:
        df: A PySpark DataFrame
        keys: A col or list of cols from the DataFrame
        probabilities: The list of probabilities to be shown in the output DataFrame
        relative_error: The relative target precision. For more information, check PySpark's
            documentation about `approxQuantile` (https://spark.apache.org/docs/latest/api/python/
            reference/pyspark.sql/api/pyspark.sql.DataFrame.approxQuantile.html). Defaults to 0.
            to obtain exact quantiles, but if the operation is too expensive you can increase this
            value (although the quantile precision will diminish)

    Returns:
        A list containing the value for each probability.
    """
    keys_count_df = get_keys_records_df(df, keys)
    return keys_count_df.approxQuantile(
        col="count", probabilities=probabilities, relativeError=relative_error
    )


def plot_partition_records_histogram(df: DataFrame, bins: int):
    """
    Helper that plots a histogram of the number of records per partitions. It can help us to
    determine if a `repartition` operation has generated equally distributed partitions.

    Args:
        df: A PySpark DataFrame
        bins: The number of bins
    """
    plt.hist(get_partition_records_df(df), bins)


def plot_keys_records_histogram(df: DataFrame, keys: Union[str, List[str]], bins: int):
    """
    Helper that plots a histogram of the number of records per keys. It can help us to
    determine if some key (prior to a join for example) is very imbalanced in relation to
    the others.

    Args:
        df: A PySpark DataFrame
        keys: A col or list of cols
        bins: The number of bins
    """
    plt.hist(get_keys_records_df(df, keys), bins)


def get_optimal_number_of_partitions(
    df: DataFrame,
    partition_cols: Union[str, List[str]] = None,
    df_sample_perc: float = None,
    target_size_in_bytes: int = 134_217_728,
    estimate_biggest_key_probability: float = 0.95,
    estimate_biggest_key_relative_error: float = 0.0,
):
    """
    This method calculated the optimal number of partitions for the input PySpark DataFrame `df`.

    Args:
        df: A PySpark DataFrame
        partition_cols: The columns, if provided, to partition the DataFrame by
        df_sample_perc: If provided, the sampling percentage for approximate size estimation
        target_size_in_bytes: The target size of each partition (~128MB)
        estimate_biggest_key_probability: In order to estimate the biggest key (that is, the partition cols key
            that contains the highest number of elements inside to estimate the size of the partitions).
        estimate_biggest_key_relative_error: The relative error of the `estimate_biggest_key_probability` estimation.
            Defaults to 0. (to obtain exact quantiles), but be careful with this since operation may be very expensive.

    Returns:
        The optimal number of partition for the given DataFrame
    """
    if not df_sample_perc:
        df_size_in_bytes = df_size_in_bytes_exact(df)
    else:
        print(
            f"Using sampling percentage {df_sample_perc}."
            f" Notice the DataFrame size is just an approximation"
        )
        df_size_in_bytes = df_size_in_bytes_approximate(df, sample_perc=df_sample_perc)

    if not partition_cols:
        n_partitions = math.ceil(df_size_in_bytes / target_size_in_bytes)

    else:
        partition_cols = (
            [partition_cols] if type(partition_cols) == str else partition_cols
        )

        # Calculate the number of elements per each partition cols grouping
        keys_count_df = df.groupBy(*partition_cols).count()
        n_unique_keys = keys_count_df.count()

        # In this case we will take a probability of 0.95 to assure that we can provide
        # reasonable estimates for partition sizes. Notice that, if your data is very skewed,
        # you could have problems with this method, since you'll still have very big partitions
        # that could generate OOM errors.
        n_rows_biggest_key = keys_count_df.approxQuantile(
            col="count",
            probabilities=[estimate_biggest_key_probability],
            relativeError=estimate_biggest_key_relative_error,
        )[0]

        biggest_partition_size = (df_size_in_bytes / df.count()) * n_rows_biggest_key

        # Careful here! If `biggest_partition_size` is greater than `target_size_in_bytes` you will get
        # one single partition. If the reason is that your partition is very big, then you'll have
        # OOM errors, and you should think about using broadcasting or maybe techniques such as salting.
        n_keys_per_partition = math.ceil(target_size_in_bytes / biggest_partition_size)
        n_partitions = math.ceil(n_unique_keys / n_keys_per_partition)

    return n_partitions

def count_cols(df: DataFrame):
    return len(df.columns)


def deduplicate(df: DataFrame, by_columns: Optional[List[str] or str] = None, order_by: Optional[List[str] or str] = None, desc_: bool = True) -> DataFrame:
    """

    Returns a DataFrame with duplicate rows removed based on the given columns.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame.
        by_columns (Union[str, List[str]]): A column or list of columns to group by for deduplication.
        order_by (Optional[Union[str, List[str]]]): A column or list of columns to order by before deduplication. If not
            specified, the deduplication will be performed based on the `by_columns` parameter.

    Returns:
        pyspark.sql.DataFrame: A DataFrame with duplicate rows removed.

    !!! Example "Deduplicating a DataFrame"
        This example shows how to use `deduplicate()` to remove duplicate rows from a DataFrame.

        === "Original df"
            ```python
            df = spark.createDataFrame([(1, "a"), (2, "b"), (1, "a"), (3, "c")], ["col1", "col2"])
            df.show()
            ```
            Output:
            ```
            +----+----+
            |col1|col2|
            +----+----+
            |   1|   a|
            |   2|   b|
            |   1|   a|
            |   3|   c|
            +----+----+
            ```

        === "Example 1"
            ```python
            df_dedup = deduplicate(df, "col1")
            df_dedup.show()
            ```
            Output:
            ```
            +----+----+
            |col1|col2|
            +----+----+
            |   1|   a|
            |   2|   b|
            |   3|   c|
            +----+----+
            ```

        === "Example 2"
            ```python
            df_dedup = deduplicate(df, ["col1", "col2"], order_by="col1")
            df_dedup.show()
            ```
            Output:
            ```
            +----+----+
            |col1|col2|
            +----+----+
            |   1|   a|
            |   2|   b|
            |   3|   c|
            +----+----+
            ```

    Info: Important
        - This function preserves the first occurrence of each unique row and removes subsequent duplicates.
        - If there are no duplicate rows in the DataFrame, this function returns the input DataFrame unchanged.
        - The `order_by` parameter can be used to specify a custom order for the deduplication. By default, the function
          orders by the columns specified in the `by_columns` parameter.
        - The input DataFrame should not contain null values, as these may cause unexpected behavior in the deduplication.
    """

    if count_cols(df) == 1:
        # Native spark function!
        return df.distinct()

    elif order_by is None or len(order_by) == 0:
        # Native spark function!
        df.dropDuplicates(subset=columns)

    else:
        columns = Str(by_columns).split_by_sep if isinstance(by_columns, str) else by_columns
        order_by_cols = Str(order_by) if isinstance(order_by, str) else Str(",".join(order_by))

        # Create a window specification to partition by key columns and order by order columns in descending order
        window_spec = Window.partitionBy(by_columns).orderBy(desc(order_by_cols.sep_to_comma))

        # Add a new column called "row_num" to the DataFrame based on the window specification
        df_num = df.withColumn("row_num", row_number().over(window_spec))

        # Filter the DataFrame to keep only rows where the "row_num" column equals 1
        df_dedup = df_num.filter(df_num.row_num == 1)

        return df_dedup.drop("row_num")

def validate_sql_query(spark, sql_query):
    """
    validate input sql query 
    """
    try:
        # Attempt to parse the SQL query without actually executing it
        spark.sql(sql_query)
        return True  # Query is valid
    except AnalysisException as e:
        print(f"Error in SQL query: {str(e)}")
        return False  # Query is invalid


def execute_sql_query(spark, query, table_name,show_row_count=False):
    """
    Execyte the input sql query and create a temporary table 
    """
    try:
        # Execute the SQL query and create a temporary table
        start_time = time()
        spark.sql(f"{query} AS {table_name}")
        # Execute SQL Query and create a temporary table
        # spark.sql(query).createOrReplaceTempView(table_name)

        # Cache the temporary table for faster access
        #spark.table(table_name).cache()
        end_time = time()
        execution_time = end_time - start_time

        if show_row_count:
            # calculate the row count
            row_count = spark.table(table_name).count()
            print(f"Table '{table_name}' created and cached. Row count: {row_count}")
        else:
            print(f"Table '{table_name}' created and cached.")
        
        print(f"Query execution time: {execution_time:.2f} seconds")
        return True
        
    except AnalysisException as e:
        print(f"Error executing the SQL query: {str(e)}")
        return False

def get_spark_session(job_name):
    """
    Get an existing Spark session or create a new one with the given job name.

    :param job_name: The name for the Spark job.
    :return: SparkSession
    """
    try:
        # Attempt to get an existing Spark session
        spark = SparkSession.builder.appName(job_name).enableHiveSupport().getOrCreate()
        return spark

    except Exception as e:
        # If no existing Spark session is available, create a new one
        spark = SparkSession.builder.appName(job_name).enableHiveSupport().getOrCreate()
        return spark
    

if __name__ == "__main__":
    job_name = "MySparkJob"

    # Get or create the Spark session
    spark = get_spark_session(job_name)

    # List of SQL statements to execute
    sql_queries = [
        {
            "query": "SELECT column1, column2 FROM table1 WHERE condition1",
            "table_name": "temp_table1"
        },
        {
            "query": "SELECT column3, column4 FROM table2 WHERE condition2",
            "table_name": "temp_table2"
        },
        {
            "query": "SELECT * FROM table3",
            "table_name": "temp_table3"
        }
        # Add more SQL queries and table names as needed
    ]

    # Execute each SQL statement and create temporary tables
    for query_info in sql_queries:
        query = query_info["query"]
        table_name = query_info["table_name"]
        if not execute_sql_query(spark, query, table_name):
            # Handle errors or log failures here
            continue

    # Join the temporary tables based on a common condition
    join_condition = "temp_table1.column1 = temp_table2.column3 AND temp_table1.column2 = temp_table3.column4"
    joined_result = spark.sql(f"SELECT * FROM temp_table1 JOIN temp_table2 ON {join_condition} JOIN temp_table3 ON {join_condition}")

    # Show or process the final result
    joined_result.show()

    # Stop the Spark session
    spark.stop()

# -------------
'''
from pyspark.sql.functions import expr, lit
from pyspark.sql.utils import AnalysisException
import sys,logging
from datetime import datetime
from typing import List, Optional, Union
from datetime import datetime
import re
from pyspark.sql import types as T
from pyspark.sql import SparkSession, DataFrame, Window
import pyspark.sql.functions as F 
from pyspark.sql.functions import desc, row_number
import sys,logging
from datetime import datetime


# Logging configuration
formatter = logging.Formatter('[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class ArenaSparkSQLManager:
    """Puts together all Spark features provided by Arena Spark SQL library.

    """
    def __init__(self, mode: str, **kwargs) -> None:
        # Cleaning up the mode string attribute to apply validations
        self.mode = mode.strip().lower()
        if self.mode == "default":
            logger.info("Create a SparkSession object")
            self.spark = SparkSession.builder.getOrCreate()

            # Logging initialization message
            logger.info("Successfully initialized Arena Spark SQL with default "
                        "operation mode. You can know use the Arena Spark SQL "
                        "features to improve your Spark application.")
        else:
           raise ValueError(f"Invalid value for operation mode (mode={mode})."
                             "Acceptable values are 'default'.")


    @staticmethod
    def run_spark_sql_pipeline(
        spark_session: SparkSession,
        spark_sql_pipeline: list
    ) -> DataFrame:
        """
        Providing a way to run multiple SparkSQL queries in sequence.

        """

        # Applying some validations on the sql pipeline argument
        required_keys = ["step", "query"]

        # Getting a list with keys of all inner dicts from sql pipeline list
        inner_pipe_keys = [list(pipe.keys()) for pipe in spark_sql_pipeline]

        # Checking if the required arguments are in all inner dict keys
        for inner_key in inner_pipe_keys:
            if not all(key in inner_key for key in required_keys):
                raise ValueError("Invalid value for spark_sql_pipeline "
                                 "argument. Please, check if all inner "
                                 "dictionaries have the 'step' and the "
                                 "'query' required keys defined accordingly. "
                                 "The main reason of this validation is "
                                 "that without the 'step' key, the method "
                                 "won't be able to sort and set the execution "
                                 "order. Without the 'query' key, the method "
                                 "won't be able to run any SparkSQL query")

        # Checking if all step keys are integers
        step_types_validation = list(set(
            [isinstance(pipe["step"], int) for pipe in spark_sql_pipeline]
        ))[0]
        if not step_types_validation:
            raise ValueError("Invalid value for spark_sql_pipeline "
                             "argument. Please check if all inner "
                             "dictionaries have the 'step' key defined as "
                             "integer values. If any 'step' key for a given "
                             "step is defined with a non integer number, the "
                             "method won't be able to sort the steps and set "
                             "the execution order accordingly")

        # Going to the method: sorting the steps in an ascending order
        sorted_spark_sql_pipeline = sorted(
            spark_sql_pipeline,
            key=lambda x: x["step"]
        )

        # Iterating over the pipeline elements to execute the statements
        for pipe in sorted_spark_sql_pipeline:
            # Executing the queries in sequence
            df_step = spark_session.sql(pipe["query"])

            # Creating a temporary view with the step result (if applicable)
            if "create_temp_tbl" not in pipe\
                    or bool(pipe["create_temp_tbl"]):
                # Assigning a name for result temporary view
                if "temp_tbl_name" not in pipe\
                        or pipe["temp_tbl_name"].strip().lower() == "auto":
                    temp_view_name = "step_" + str(pipe["step"])
                else:
                    temp_view_name = pipe["temp_tbl_name"]

                # Creating the temporary view
                df_step.createOrReplaceTempView(temp_view_name)
                logger.info(f'Created Temp Table: {temp_view_name} ')
                
                # Cache the temporary table for faster access
                spark_session.table(temp_view_name).cache()
                logger.info(f'Cached Temp Table: {temp_view_name} ')


        # Returning the DataFrame from the last step
        return df_step
    
# Creating a spark manager object to develop Spark apps anywhere
spark_manager = ArenaSparkSQLManager(mode="default")

# create dataset-2
df_bakery = spark_manager.spark.read \
    .format('csv') \
    .option('header', 'true') \
    .option('delimiter', ',') \
    .option('inferSchema', 'true') \
    .load('BreadBasket_DMS.csv')

df_bakery.createOrReplaceTempView('tbl_bakery')

# create a datasets-1
df_customer = spark_manager.spark.read \
    .format('csv') \
    .option('header', 'true') \
    .option('delimiter', ',') \
    .option('inferSchema', 'true') \
    .load('customer.csv')

df_customer.createOrReplaceTempView('tbl_customer')


# prepare the SQL to execute 
spark_sql_pipeline = [
            {
                "step": 1,
                "query": "SELECT date, Transaction, time FROM tbl_bakery WHERE item='Coffee'",
                "create_temp_tbl" : True,
                "temp_tbl_name": "auto"
            },
            {
                "step": 2,
                "query":"SELECT step_1.date, sum(step_1.transaction)  FROM step_1 GROUP BY step_1.date ",
                "create_temp_tbl": True,
                "temp_tbl_name": "auto"
            }
        ]

# Running the pipeline
df_prep = spark_manager.run_spark_sql_pipeline(
    spark_session=spark_manager.spark,
    spark_sql_pipeline=spark_sql_pipeline
)

# Showing a sample
df_prep.show(5, truncate=False)



# RDBMS ingestion function
def init_spark(config, app=None, use_session=False):
    import os
    import sys
    from glob import glob

    if 'spark-home' in config:
        os.environ['SPARK_HOME'] = config['spark-home']

    if 'spark-conf-dir' in config:
        os.environ['SPARK_CONF_DIR'] = config['spark-conf-dir']

    if 'pyspark-python' in config:
        # Set python interpreter on both driver and workers
        os.environ['PYSPARK_PYTHON'] = config['pyspark-python']

    if 'yarn-conf-dir' in config:
        # Hadoop YARN configuration
        os.environ['YARN_CONF_DIR'] = config['yarn-conf-dir']

    if 'spark-classpath' in config:
        # can be used to set external folder with Hive configuration
        # e. g. spark-classpath='/etc/hive/conf.cloudera.hive1'
        os.environ['SPARK_CLASSPATH'] = config['spark-classpath']

    submit_args = []

    driver_mem = config.get('spark-prop.spark.driver.memory', None)
    if driver_mem is not None:
        submit_args.extend(["--driver-memory", driver_mem])

    driver_cp = config.get('spark-prop.spark.driver.extraClassPath', None)
    if driver_cp is not None:
        submit_args.extend(["--driver-class-path", driver_cp])

    driver_java_opt = config.get('spark-prop.spark.driver.extraJavaOptions', None)
    if driver_java_opt is not None:
        submit_args.extend(["--driver-java-options", driver_java_opt])

    jars = config.get('jars', None)
    if jars is not None:
        if isinstance(jars, str):
            jars = [jars]
        submit_args.extend(["--jars", ','.join(jars)])

    mode_yarn = config['spark-prop.spark.master'].startswith('yarn')

    if mode_yarn:
        # pyspark .zip distribution flag is set only if spark-submit have master=yarn in command-line arguments
        # see spark.yarn.isPython conf property setting code
        # in org.apache.spark.deploy.SparkSubmit#prepareSubmitEnvironment
        submit_args.extend(['--master', 'yarn'])

    # pyspark .zip distribution flag is set only if spark-submit have pyspark-shell or .py as positional argument
    # see spark.yarn.isPython conf property setting code
    # in org.apache.spark.deploy.SparkSubmit#prepareSubmitEnvironment
    submit_args.append('pyspark-shell')

    os.environ['PYSPARK_SUBMIT_ARGS'] = ' '.join(submit_args)

    spark_home = os.environ['SPARK_HOME']
    spark_python = os.path.join(spark_home, 'python')
    pyspark_libs = glob(os.path.join(spark_python, 'lib', '*.zip'))
    sys.path.extend(pyspark_libs)

    virtualenv_reqs = config['spark-prop'].get('spark.pyspark.virtualenv.requirements', None)
    if use_session:
        from pyspark.sql import SparkSession

        builder = SparkSession.builder.appName(app or config['app'])

        if mode_yarn:
            builder = builder.enableHiveSupport()

        for k, v in prop_list(config['spark-prop']).items():
            builder = builder.config(k, v)

        ss = builder.getOrCreate()
        if virtualenv_reqs is not None:
            ss.addFile(virtualenv_reqs)
        return ss
    else:
        from pyspark import SparkConf, SparkContext
        conf = SparkConf()
        conf.setAppName(app or config['app'])
        props = [(k, str(v)) for k, v in prop_list(config['spark-prop']).items()]
        conf.setAll(props)
        sc = SparkContext(conf=conf)
        if virtualenv_reqs is not None:
            sc.addFile(virtualenv_reqs)
        return sc


def init_session(config, app=None, return_context=False, overrides=None, use_session=False):
    import os
    from pyhocon import ConfigFactory, ConfigParser

    if isinstance(config, str):
        if os.path.exists(config):
            base_conf = ConfigFactory.parse_file(config, resolve=False)
        else:
            base_conf = ConfigFactory.parse_string(config, resolve=False)
    elif isinstance(config, dict):
        base_conf = ConfigFactory.from_dict(config)
    else:
        base_conf = config

    if overrides is not None:
        over_conf = ConfigFactory.parse_string(overrides)
        conf = over_conf.with_fallback(base_conf)
    else:
        conf = base_conf
        ConfigParser.resolve_substitutions(conf)

    res = init_spark(conf, app, use_session)

    if use_session:
        return res
    else:
        mode_yarn = conf['spark-prop.spark.master'].startswith('yarn')

        if mode_yarn:
            from pyspark.sql import HiveContext
            sqc = HiveContext(res)

            if 'hive-prop' in conf:
                for k, v in prop_list(conf['hive-prop']).items():
                    sqc.setConf(k, str(v))
        else:
            from pyspark.sql import SQLContext
            sqc = SQLContext(res)

        if return_context:
            return res, sqc
        else:
            return sqc
        
def jdbc_load(
    sqc,
    query,
    conn_params,
    partition_column=None,
    num_partitions=10,
    lower_bound=None,
    upper_bound=None,fetch_size=10000000
):
    import re
    if re.match(r'\s*\(.+\)\s+as\s+\w+\s*', query):
        _query = query
    else:
        _query = '({}) as a'.format(query)

    conn_params_base = dict(conn_params)
    if partition_column and num_partitions and num_partitions > 1:
        if lower_bound is None or upper_bound is None:
            min_max_query = '''
              (select max({part_col}) as max_part, min({part_col}) as min_part
                 from {query}) as g'''.format(part_col=partition_column, query=_query)
            max_min_df = sqc.read.load(dbtable=min_max_query, **conn_params_base)
            tuples = max_min_df.rdd.collect()
            lower_bound = str(tuples[0].max_part)
            upper_bound = str(tuples[0].min_part)
        conn_params_base['fetchSize'] = str(fetch_size)
        conn_params_base['partitionColumn'] = partition_column
        conn_params_base['lowerBound'] = lower_bound
        conn_params_base['upperBound'] = upper_bound
        conn_params_base['numPartitions'] = str(num_partitions)
    sdf = sqc.read.load(dbtable=_query, **conn_params_base)
    return sdf

def partition_iterator(sdf):
    import pyspark.sql.functions as F
    sdf_part = sdf.withColumn('partition', F.spark_partition_id())
    sdf_part.cache()
    for part in range(sdf.rdd.getNumPartitions()):
        yield sdf_part.where(F.col('partition') == part).drop('partition').rdd.toLocalIterator()

def proportion_samples(sdf, proportions_sdf, count_column='rows_count'):
    '''Load huge tables from Hive slightly faster than over toPandas in Spark
    Parameters
    ----------
    sdf : spark Dataframe to sample from
    proportions_sdf : spark Dataframe with counts to sample from sdf
    count_column: column name with counts, other columns used as statifiers
 
    Returns
    ----------
    sampled : spark Dataframe with number of rows lesser or equal proportions_sdf for each strata
    '''
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window
    groupers = [c for c in proportions_sdf.columns if c != count_column]

    sampled = sdf.join(proportions_sdf, groupers, how='inner').withColumn(
        'rownum',
        F.rowNumber().over(Window.partitionBy(groupers))
    ).filter(
        F.col('rownum') <= F.col(count_column)
    ).drop(count_column).drop('rownum')
    return sampled


'''
