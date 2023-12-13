import pytest
import os
from pyspark.sql import SparkSession
from pathlib import Path
from pyspark.sql.types import StructType, StructField, IntegerType


INPUT_TEST_DATA = Path("./test/test_data/input")
BIG_FILES = set(['50_100', '50_200', '100_200', '50_1000', '100_1000', '1000_1000'])


def get_test_data_paths(only_small=True):
    skip_files = set(BIG_FILES) if only_small else set()
    for input_file in sorted(INPUT_TEST_DATA.glob("*.csv"), key=os.path.getsize):
        test_name = input_file.stem
        if test_name in skip_files:
            continue
        output_file = INPUT_TEST_DATA.parent / "output" / f"{test_name}_output.csv"

        yield input_file.as_posix(), output_file.as_posix()

@pytest.fixture
def spark(scope="session"):
    spark = SparkSession.builder.appName("pyspark unittest").getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def edge_schema():
    yield StructType([
        StructField("edge_1", IntegerType(), True),
        StructField("edge_2", IntegerType(), True),
        StructField("length", IntegerType(), True)
    ])

@pytest.fixture()
def example1_io(spark, edge_schema):
    _input = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
    ], edge_schema)

    _output = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (1, 3, 2),
        (1, 4, 3),
        (2, 4, 2)
    ], edge_schema)

    yield _input, _output

@pytest.fixture()
def example2_io(spark, edge_schema):
    _input = spark.createDataFrame([
        (1, 2, 1),
        (1, 3, 3),
        (2, 3, 1)
    ], edge_schema)

    _output = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (1, 3, 2)
    ], edge_schema)

    yield _input, _output

@pytest.fixture()
def cycle_io(spark, edge_schema):
    _input = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (3, 1, 1)
    ], edge_schema)

    _output = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (3, 1, 1),
        (1, 3, 2),
        (2, 1, 2),
        (3, 2, 2),
        (1, 1, 3),
        (2, 2, 3),
        (3, 3, 3),
    ], edge_schema)

    yield _input, _output

@pytest.fixture()
def branch_io(spark, edge_schema):
    _input = spark.createDataFrame([
        (1, 2, 1),
        (2, 4, 1),
        (2, 5, 1)
    ], edge_schema)

    _output = spark.createDataFrame([
        (1, 2, 1),
        (2, 4, 1),
        (2, 5, 1),
        (1, 4, 2),
        (1, 5, 2),
    ], edge_schema)

    yield _input, _output

@pytest.fixture()
def shortcut_io(spark, edge_schema):
    _input = spark.createDataFrame([
        (1, 2, 5),
        (2, 3, 5),
        (1, 4, 1),
        (4, 5, 1),
        (5, 3, 1),
    ], edge_schema)

    _output = spark.createDataFrame([
        (1, 2, 5),
        (2, 3, 5),
        (1, 4, 1),
        (4, 5, 1),
        (5, 3, 1),
        (1, 5, 2),
        (4, 3, 2),
        (1, 3, 3),
    ], edge_schema)

    yield _input, _output

@pytest.fixture()
def redundand_edge(spark, edge_schema):
    _input = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (1, 3, 5),
    ], edge_schema)

    _output = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (1, 3, 2),
    ], edge_schema)

    yield _input, _output

@pytest.fixture()
def kj_test_20_100(spark, edge_schema):
    _input = spark.createDataFrame([
        (0, 9, 4),
        (9, 12, 5),
        (12, 13, 1),
        (13, 11, 2),
        (11, 6, 2),
        (0, 14, 7),
        (14, 12, 3),
        (14, 4, 4),
        (4, 5, 4),
        (5, 6, 1),
        (4, 6, 5)
    ], edge_schema)

    _output_subset = spark.createDataFrame([
        (0, 6, 14)
    ], edge_schema)

    yield _input, _output_subset

@pytest.fixture()
def partial_shortcut(spark, edge_schema):
    _input = spark.createDataFrame([
        (0, 1, 5),
        (1, 2, 5),
        (2, 13, 1),
        (13, 14, 1),
        (14, 15, 1),
        (0, 3, 1),
        (3, 4, 1),
        (4, 1, 1),
    ], edge_schema)

    _output_subset = spark.createDataFrame([
        (0, 1, 3),
        (0, 15, 11),
    ], edge_schema)

    yield _input, _output_subset

@pytest.fixture()
def only_length_updates(spark, edge_schema):
    _input = spark.createDataFrame([
        (0, 1, 10),
        (0, 2, 1),
        (2, 3, 1),
        (3, 1, 1),
        (2, 1, 2),
        (0, 3, 2),
    ], edge_schema)

    _output_subset = spark.createDataFrame([
        (0, 1, 3),
    ], edge_schema)

    yield _input, _output_subset

@pytest.fixture()
def only_length_updates_2(spark, edge_schema):
    _input = spark.createDataFrame([
        (0, 1, 5),
        (0, 2, 11),
        (0, 3, 16),
        (0, 4, 21),
        (1, 2, 5),
        (1, 3, 11),
        (1, 4, 16),
        (2, 3, 5),
        (2, 4, 10),
        (3, 4, 5),
    ], edge_schema)

    _output_subset = spark.createDataFrame([
        (0, 4, 20),
    ], edge_schema)

    yield _input, _output_subset