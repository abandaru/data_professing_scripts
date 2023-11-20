from pyspark.sql import functions as F
from pyspark.sql.window import Window
import random
import string
import glob
import os
import logging
import sys
import copy
import yaml
from datetime import datetime
import time
import decimal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('spark-rdbms-ingestor')

class ConsistencyError(AssertionError):
    pass

def add_common_arguments(parser):
    parser.add_argument('-c', '--config')
    parser.add_argument('-u', '--jdbc')
    parser.add_argument('-D', '--driver')
    parser.add_argument('-U', '--username')
    parser.add_argument('-P', '--password')
    parser.add_argument('-t', '--dbtable')
    parser.add_argument('-H', '--hive-table')
    parser.add_argument('-q', '--query')
    parser.add_argument('-p', '--partition-column')
    parser.add_argument('-y', '--output-partition-columns')
    parser.add_argument('-m', '--num-partitions')
    parser.add_argument('-T', '--query-timeout')
    parser.add_argument('-F', '--fetch-size')
    parser.add_argument('-I', '--init')
    parser.add_argument('-i', '--ingestion-tag-column',
            default='dl_ingest_date')
    parser.add_argument('-s', '--storageformat', default='parquet')
    parser.add_argument('-O', '--overwrite', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--consistency-threshold', type=int, default=0)
    parser.add_argument('--consistency-retry', type=int, default=3)
    parser.add_argument('--consistency-retry-delay', type=int, default=60)

    
def validate_common_args(args):
    if not args.jdbc:
        print("-u/--jdbc is required")
        sys.exit(1)
    if args.dbtable and args.query:
        print('Either -t/--dbtable or -q/--query shall be specified, but not both')
        sys.exit(1)
    if not args.dbtable and not args.query:
        print('Either -t/--dbtable or -q/--query must be specified')
        sys.exit(1)
    if not args.dbtable and not args.hive_table:
        print('-T/--hive-table is required when using with -q/--query')
        sys.exit(1)
    
    if ((args.num_partitions and not args.partition_column) or
       (args.partition_column and not args.num_partitions)):
           print('-m/--num-partitions and -p/--partition-column must '
            'be specified together')
           sys.exit(1)
    
    if ((args.username and  not args.password) or
       (args.password and not args.username)):
           print('-U/--username and -P/--password must '
            'be specified together')
           sys.exit(1)
    
def base_conn_from_args(spark, args):
    conn = spark.read.format('jdbc').option('url', args.jdbc)
    if args.driver:
        conn = conn.option('driver', args.driver)
    if args.username:
        conn = conn.option('user', args.username)
    if args.password:
        conn = conn.option('password', args.password)
    if args.query_timeout:
        conn = conn.option('queryTimeout', args.query_timeout)
    if args.fetch_size:
        conn = conn.option('fetchSize', args.fetch_size)
    if args.init:
        conn = conn.option('sessionInitStatement', args.init)

    jdbc_uri = args.jdbc
    if jdbc_uri.startswith('oracle') or jdbc_uri.startswith('jdbc:oracle'):
        conn = (conn
             .option('oracle.jdbc.mapDateToTimestamp', 'false')
             .option('sessionInitStatement', """BEGIN execute immediate 'ALTER SESSION SET NLS_TIMESTAMP_FORMAT="YYYY-MM-DD HH24:MI:SS.FF"'; 
                                                      execute immediate 'ALTER SESSION SET NLS_DATE_FORMAT="YYYY-MM-DD"';
                                                END;"""))
    
    return conn


class Args(object):
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, key):
        return None

def parse_args(parser, argv):
    args = parser.parse_args(argv)
    args = args_from_config(args)
    args = parser.parse_args(argv, namespace=args)
    validate_common_args(args)
    return args

def args_from_config(args):
    if not args.config:
        return args
    with open(args.config, 'r') as cf:
        config = yaml.safe_load(cf)
    return Args(**config)

def get_db_type(jdbc_uri):
    uri = jdbc_uri.lower()
    if uri.startswith('jdbc:'):
        uri = uri[5:]
    if uri.startswith('oracle'):
        return 'oracle'
    if uri.startswith('sqlserver'):
        return 'mssql'
    return 'ansi'

def get_jdbc_schema(spark, args):
    conn = spark._jvm.java.sql.DriverManager.getConnection(args.jdbc, args.username, args.password)
    stmt = conn.createStatement()
    db_type = get_db_type(args.jdbc)
    if db_type == 'oracle':
        query = 'select * from (%s) q where row_num < 1' % (args.dbtable or args.query)
    elif db_type == 'mssql':
        query = 'select top 1 * from (%s) q' % (args.dbtable or args.query)
    else:
        query = 'select * from (%s) q limit 1' % (args.dbtable or args.query)
    resultset = stmt.executeQuery(query)
    schema = {}
    metadata = resultset.getMetaData()
    for i in range(metadata.getColumnCount()):
        idx = i+1
        name = metadata.getColumnName(idx)
        dtype = metadata.getColumnTypeName(idx)
        schema[name] = dtype
    return schema

def get_column(df, column):
    for field in df.schema.fields:
        if field.name.lower() == column.lower():
            return field
    raise KeyError('unknown field')

def conn_from_args(spark, args, query=None, use_partitioning=True):
    if args.verbose:
        spark.sparkContext.setLogLevel("INFO")
    else:
        spark.sparkContext.setLogLevel("WARN")

    conn = base_conn_from_args(spark, args)

    if query or args.query:
        conn = conn.option('query', query or args.query)
    elif args.dbtable:
        conn = conn.option('dbtable', args.dbtable)
    else:
        raise AssertionError('Neither dbtable nor query are available')
    
    if use_partitioning and args.partition_column and args.num_partitions:
        cdfx = conn.load()
        colschema = get_column(cdfx, args.partition_column)
        query = 'select min(%s) as lower_bound, max(%s) as upper_bound from %s' % (args.partition_column, args.partition_column, args.dbtable)
        pushdownConn = base_conn_from_args(spark, args)
        dfx = pushdownConn.option('query', query).load()
        log.info('Getting lower and upper bound of %s' % args.partition_column)
        bounds = dfx.take(1)[0]
        lower_bound = bounds[0]
        upper_bound = bounds[1]

        if isinstance(lower_bound, decimal.Decimal):
            if colschema.dataType.scale == 0:
                lower_bound = int(lower_bound)
            else:
                lower_bound = float(lower_bound)
        if isinstance(upper_bound, decimal.Decimal):
            if colschema.dataType.scale == 0:
                upper_bound = int(upper_bound)
            else:
                upper_bound = float(upper_bound)
        log.info('Lower bound = %s' % lower_bound)
        log.info('Upper bound = %s' % upper_bound)
        
        if lower_bound is not None:
            if upper_bound is None:
                raise AssertionError("Lower bound = %s but upper bound is None" % lower_bound)
        else:
            if upper_bound is not None:
                raise AssertionError("Upper bound = %s but lower bound is None" % upper_bound)
        
        if lower_bound is not None and upper_bound is not None:
            conn = (conn.option('partitionColumn', args.partition_column)
                .option('numPartitions', str(args.num_partitions))
                .option('lowerBound', str(lower_bound))
                .option('upperBound', str(upper_bound)))

    return conn

_marker = []

def get_source_count(spark, args):
    conn = base_conn_from_args(spark, args)
    if args.dbtable:
        query = 'select count(1) as rowc from %s' % args.dbtable
    else:
        query = 'select count(1) as rowc from (%s) as tbl' % args.query

    df = conn.option('query', query).load()
    return df.take(1)[0][0]

def full_ingestion_with_retry(args, spark, *sargs, **kwargs):

    success = False

    for i in range(args.consistency_retry):
        log.info('[%s/%s] Ingesting data' % (i, args.consistency_retry))
        source_count = get_source_count(spark, args)

        ingested_count = full_ingestion(spark, *sargs, **kwargs)

        if source_count == 0 and ingested_count == 0:
            success = True
            break

        if source_count:
            diff = (abs(float(source_count - ingested_count)) / float(source_count)) * 100

            if not (diff > args.consistency_threshold):
                success = True
                break

        log.warn("[%s/%s] Consistency check failed.")
        log.warn("Sleeping for %s secs" % args.consistency_retry_delay)
        time.sleep(args.consistency_retry_delay)

    if not success:
        raise ConsistencyError("Source rows = %s, Ingested rows = %s" % (source_new_rows, ingested_count))


    return ingested_count
        
def full_ingestion(spark, df, hive_db, hive_tbl, drop=False,
        storageformat='parquet', ingestion_tag_column='dl_ingest_date',
        output_partitions=_marker):
    
    output_partitions = output_partitions or []

    db, tbl = hive_db, hive_tbl
    
    df = df.withColumn(ingestion_tag_column, F.lit(datetime.now().strftime('%Y%m%dT%H%M%S')))
    log.info('Ingesting into spark persistence cache')
    df = df.persist()
    log.info('Persistence done')
    df.createOrReplaceTempView('import_tbl')
    new_rows = df.count()
    log.info("Ingesting %s new rows" % new_rows)
    
    log.info('Importing %s' % tbl)
    spark.sql('create database if not exists %s' % db)
    if drop:
       spark.sql('drop table if exists %s.%s' % (db, tbl))
    spark.sql('create table if not exists %s.%s stored as %s as select * from import_tbl limit 0' % (db, tbl, storageformat))

    tbl_df = spark.sql('select * from %s.%s' % (db, tbl))
    if len(df.columns) != len(tbl_df.columns):
        raise AssertionError(
            'column count in %s.%s (%s) does not match column count in dataframe (%s)' % (
                db, tbl, str(tbl_df.columns), str(df.columns))
        )
    df = df.select(*tbl_df.columns)
    df.write.format(storageformat).insertInto('%s.%s' % (db, tbl), overwrite=True)    
    log.info('.. DONE')

    return new_rows

def incremental_append_metadata(spark, df, hive_db, hive_tbl, incremental_column):
    db, tbl = hive_db, hive_tbl
    incremental_exists = False
    last_value = None
    if db.lower() in [d.name.lower() for d in spark.catalog.listDatabases()]:
        tables = [t.name.lower() for t in spark.catalog.listTables(db)]
        if tbl.lower() in tables:
            incremental_exists = True
   
    last_value = None
    if incremental_exists and incremental_column:
        last_value = spark.sql('select max(%s) from %s.%s' % (incremental_column, db, tbl)).take(1)[0][0]

    return {
        'incremental_exists': incremental_exists,
        'last_value': last_value
    }

def incremental_append_ingestion_with_retry(args, spark,  df, hive_db, hive_tbl, 
                                            incremental_column,
                                            *sargs, **kwargs):
    meta = incremental_append_metadata(spark, df, hive_db, hive_tbl, incremental_column)

    if args.dbtable:
        query = 'select count(1) as rowc from %s' % (args.dbtable)
    else:
        query = 'select count(1) as rowc from (%s) as tbl' % (args.query)

    dbtype = get_db_type(args.jdbc)
    # TODO: add dialect support
    if incremental_column and meta['last_value']:
        colinfo = get_column(df, incremental_column)
        if colinfo.dataType.typeName() == 'timestamp':
            if dbtype == 'oracle':
                query += " where (%s > to_timestamp('%s', 'YYYY-MM-DD HH24:MI:SS.FF'))" % (incremental_column, meta['last_value'])
            else:
                query += " where (%s > '%s')" % (incremental_column, meta['last_value'])
        elif colinfo.dataType.typeName() in ['decimal', 'float']:
            query += " where (%s > %f)" % (incremental_column, meta['last_value'])
        elif colinfo.dataType.typeName() in ['integer']:
            query += " where (%s > %d)" % (incremental_column, meta['last_value'])
        else:
            query += " where (%s > '%s')" % (incremental_column, meta['last_value'])

    success = False
    for i in range(args.consistency_retry):
        log.info('[%s/%s] Ingesting data' % (i+1, args.consistency_retry))
        conn = base_conn_from_args(spark, args)
        sdf = conn.option('query', query).load()
        source_new_rows = sdf.take(1)[0][0]

        if source_new_rows == 0:
            success = True
            new_rows = 0
            log.info("No new data in source")
            break

        new_rows = incremental_append_ingestion(spark, df, hive_db, hive_tbl,
                                            incremental_column, *sargs, **kwargs)

        diff = (abs(float(source_new_rows - new_rows))/float(source_new_rows)) * 100
        if not (diff > args.consistency_threshold):
            success = True
            break

        last_ingest_date = spark.sql('select max(%s) from %s.%s' % (kwargs['ingestion_tag_column'], hive_db, hive_tbl)).collect()[0][0]
        log.warn("[%s/%s] Consistency check failed. Deleting partition %s=%s" % (
            i+1, args.consistency_retry, kwargs['ingestion_tag_column'], last_ingest_date))
        spark.sql('alter table %s.%s drop partition (%s="%s") purge' % (
            hive_db, hive_tbl, kwargs['ingestion_tag_column'], 
            last_ingest_date))
        df = conn_from_args(spark, args).load()
        df.unpersist()
        log.warn("Sleeping for %s secs" % args.consistency_retry_delay)
        time.sleep(args.consistency_retry_delay)

    if not success:
        raise ConsistencyError("Source rows = %s, Ingested rows = %s" % (source_new_rows, new_rows))

    return new_rows

def incremental_append_ingestion(spark, df, hive_db, hive_tbl,
        incremental_column, last_value=None, storageformat='parquet',
        ingestion_tag_column='dl_ingest_date',
        output_partitions=_marker):

    output_partitions = output_partitions or [ingestion_tag_column]

    db, tbl = hive_db, hive_tbl

    meta = incremental_append_metadata(spark, df, hive_db, hive_tbl, incremental_column)

    incremental_exists = meta['incremental_exists']

    if incremental_exists and incremental_column and not last_value:
        last_value = meta['last_value']

    if incremental_column and last_value:
        df = df.where(F.col(incremental_column) > F.lit(last_value))

    df = df.withColumn(ingestion_tag_column, F.lit(datetime.now().strftime('%Y%m%dT%H%M%S')))
    df = df.persist()
    new_rows = df.count()
    log.info("Ingesting %s new rows" % new_rows)

    if not incremental_exists:
        log.info('Importing %s' % tbl)
        spark.sql('create database if not exists %s' % db)
        df.write.mode('overwrite').format(storageformat).partitionBy(*output_partitions).saveAsTable('%s.%s' % (db, tbl))
        log.info('.. DONE')
    else:
        log.info('Importing incremental %s' % tbl)
        df.write.mode('append').format(storageformat).partitionBy(*output_partitions).saveAsTable('%s.%s' % (db, tbl))
        log.info('.. DONE')

    return new_rows


def incremental_merge_metadata(spark, hive_db, hive_tbl, last_modified_column, incremental_column):

    db, tbl = hive_db, hive_tbl
    incremental_exists = False
    incremental_tbl = '%s_incremental' % tbl
    if db.lower() in [d.name.lower() for d in spark.catalog.listDatabases()]:
        tables = [t.name.lower() for t in spark.catalog.listTables(db)]
        if incremental_tbl.lower() in tables:
            incremental_exists = True

    last_modified = None
    last_modified_type = None
    if incremental_exists and last_modified_column:
        last_modified_df = spark.sql('select max(%s) from %s.%s' % (last_modified_column, db, incremental_tbl))
        last_modified = last_modified_df.take(1)[0][0]
        last_modified_type = last_modified_df.schema[0].dataType.typeName()

    last_value = None
    last_value_type = None
    if incremental_exists and incremental_column:
        last_value_df = spark.sql('select max(%s) from %s.%s' % (incremental_column, db, incremental_tbl))
        last_value = last_value_df.take(1)[0][0]
        last_value_type = last_value_df.schema[0].dataType.typeName()

    return {
        'incremental_tbl': incremental_tbl,
        'incremental_exists': incremental_exists,
        'last_modified': last_modified,
        'last_modified_type': last_modified_type,
        'last_value': last_value,
        'last_value_type': last_value_type,
    }

def db_value(db_type, value, type_name):
    if value is None:
        return None
    if db_type == 'mssql':
        if type_name == 'timestamp':
            return "convert(datetime, '%s')" % value.isoformat()[:23]
    return "'%s'" % value

def incremental_merge_ingestion_with_retry(args, spark, df, hive_db, hive_tbl, key_columns,
        last_modified_column, *sargs, **kwargs):
    meta = incremental_merge_metadata(spark, hive_db, hive_tbl, last_modified_column, kwargs['incremental_column'])

    if args.dbtable:
        query = 'select count(1) as rowc from %s' % (args.dbtable)
    else:
        query = 'select count(1) as rowc from (%s) as tbl' % (args.query)

    db_type = get_db_type(args.jdbc)

    incremental_exists = meta['incremental_exists']
    incremental_column = kwargs.get('incremental_column', None)
    last_modified = kwargs.get('last_modified', None)
    last_value = kwargs.get('last_value', None)
    incremental_tbl = meta['incremental_tbl']

    if last_modified_column and not last_modified and incremental_exists:
        last_modified = db_value(db_type, meta['last_modified'], meta['last_modified_type'])

    if incremental_column and not last_value and incremental_exists:
        last_value = db_value(db_type, meta['last_value'], meta['last_value_type'])

    filters = {
        'incremental_column': incremental_column,
        'last_value': last_value,
        'last_modified_column': last_modified_column,
        'last_modified': last_modified
    }

    # TODO: add dialect support

    if (last_modified_column and last_modified 
            and incremental_column and last_value):

        query += (" where ((%(incremental_column)s > %(last_value)s) "
                  "or (%(last_modified_column)s > %(last_modified)s))") % filters
        #df = df.where((F.col(incremental_column) > F.lit(last_value)) |
        #        (F.col(last_modified_column) > F.lit(last_modified)))
    elif last_modified_column and last_modified:
        #df = df.where(F.col(last_modified_column) > F.lit(last_modified))
        query += " where (%(last_modified_column)s > %(last_modified)s)" % filters
    elif incremental_column and last_value:
        # df = df.where(F.col(incremental_column) > F.lit(last_value))
        query += " where (%(incremental_column)s > %(last_value)s)" % filters

    success = False
    for i in range(args.consistency_retry):
        log.info('[%s/%s] Ingesting data' % (i, args.consistency_retry))
        conn = base_conn_from_args(spark, args)
        log.warn('Executing %s' % query)
        sdf = conn.option('query', query).load()
        source_new_rows = sdf.take(1)[0][0]

        if source_new_rows == 0:
            new_rows = 0
            log.info("No new rows in source")
            success = True
            break

        new_rows = incremental_merge_ingestion(spark, df, hive_db, hive_tbl, key_columns,
            last_modified_column, *sargs, **kwargs)

        diff = (abs(float(source_new_rows - new_rows))/float(source_new_rows)) * 100
        if not (diff > args.consistency_threshold):
            success = True
            break

        last_ingest_date = spark.sql('select max(%s) from %s.%s' % (kwargs['ingestion_tag_column'], hive_db, incremental_tbl)).collect()[0][0]
        log.warn("[%s/%s] Consistency check failed. Deleting partition %s=%s" % (
            i+1, args.consistency_retry, kwargs['ingestion_tag_column'], last_ingest_date))
        spark.sql('alter table %s.%s drop partition (%s="%s") purge' % (
            hive_db, incremental_tbl, kwargs['ingestion_tag_column'], 
            last_ingest_date))
        df = conn_from_args(spark, args).load()
        df.unpersist()
        log.warn("Sleeping for %s secs" % args.consistency_retry_delay)
        time.sleep(args.consistency_retry_delay)

    if not success:
        raise ConsistencyError("Source rows = %s, Ingested rows = %s" % (source_new_rows, new_rows))

    return new_rows



def incremental_merge_ingestion(spark, df, hive_db, hive_tbl, key_columns,
        last_modified_column, last_modified=None, 
        incremental_column=None, last_value=None,
        deleted_column=None, scratch_db='spark_scratch', 
        storageformat='parquet', ingestion_tag_column='dl_ingest_date',
        output_partitions=_marker):

    output_partitions = output_partitions or []

    db, tbl = hive_db, hive_tbl

    meta = incremental_merge_metadata(spark, hive_db, hive_tbl, last_modified_column, incremental_column)

    incremental_exists = meta['incremental_exists']
    incremental_tbl = meta['incremental_tbl']

    if last_modified_column and not last_modified and incremental_exists:
        last_modified = meta['last_modified']

    if incremental_column and not last_value and incremental_exists:
        last_value = meta['last_value']

    if (last_modified_column and last_modified 
            and incremental_column and last_value):
        df = df.where((F.col(incremental_column) > F.lit(last_value)) |
                (F.col(last_modified_column) > F.lit(last_modified)))
    elif last_modified_column and last_modified:
        df = df.where(F.col(last_modified_column) > F.lit(last_modified))
    elif incremental_column and last_value:
        df = df.where(F.col(incremental_column) > F.lit(last_value))

    df = df.withColumn(ingestion_tag_column, F.lit(datetime.now().strftime('%Y%m%dT%H%M%S')))
    df = df.persist()
    new_rows = df.count()
    log.info("Ingesting %s new rows" % new_rows)

    if not incremental_exists:
        log.info('Importing %s' % tbl)
        spark.sql('create database if not exists %s' % db)
        df.write.mode('overwrite').format(storageformat).partitionBy(ingestion_tag_column).saveAsTable('%s.%s' % (db, incremental_tbl))
        log.info('.. DONE')
    else:
        log.info('Importing incremental %s' % tbl)
        df.write.mode('append').format(storageformat).partitionBy(ingestion_tag_column).saveAsTable('%s.%s' % (db, incremental_tbl))
        log.info('.. DONE')

    df = spark.sql('select * from %s.%s' % (db, incremental_tbl))

    # reconcile and select latest record
    row_num_col = 'row_num_%s' % ''.join(random.sample(string.ascii_lowercase, 6))
    windowSpec = (
        Window.partitionBy(*key_columns)
              .orderBy(F.col(last_modified_column).desc())
    )
    reconcile_df = df.select(
        F.row_number().over(windowSpec).alias(row_num_col),
        *df.columns
    )
    reconcile_df = reconcile_df.where(F.col(row_num_col) == F.lit(1)).drop(row_num_col)
    if deleted_column:
        reconcile_df = reconcile_df.where(F.col(deleted_column).isNull())
    
    reconcile_df.createOrReplaceTempView('import_tbl')
    
    log.info('Importing/Updating %s' % tbl)
    
    df = reconcile_df.persist()
    temp_table = 'temp_table_%s' % ''.join(random.sample(string.ascii_lowercase, 6))
    
    # materialize reconciled data
    df.createOrReplaceTempView(temp_table)
    spark.sql('create database if not exists %s' % scratch_db)
    writer = df.write.mode('overwrite').format(storageformat)
    if output_partitions:
        writer = writer.partitionBy(*output_partitions)
    try:
        writer.saveAsTable('%s.%s_persist' % (scratch_db, temp_table))
    except Exception as e:
        spark.sql('drop table %s.%s_persist' %  (scratch_db, temp_table))
        raise e
    
    # move materialized data to destination table
    dfx = spark.sql('select * from %s.%s_persist' % (scratch_db, temp_table))
    spark.sql('create table if not exists %s.%s like %s.%s_persist'
            % (db, tbl, scratch_db, temp_table))

    tbl_df = spark.sql('select * from %s.%s' % (db, tbl))
    if len(dfx.columns) != len(tbl_df.columns):
        raise AssertionError(
            'column count in %s.%s (%s) does not match column count in dataframe (%s)' % (
                db, tbl, str(tbl_df.columns), str(dfx.columns))
        )
    dfx = dfx.select(*tbl_df.columns)
    writer = dfx.write.format(storageformat)
    try:
        writer.insertInto('%s.%s' % (db, tbl), overwrite=True)
    except Exception as e:
        raise e
    finally:
        spark.sql('drop table %s.%s_persist' %  (scratch_db, temp_table))
    log.info('.. DONE')
    
    return new_rows


