import pytest
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.testing.utils import assertDataFrameEqual
from pathlib import Path

from lib.utils import linear_paths_algorithm


INPUT_TEST_DATA = Path("./test/test_data/input")


def get_test_data_paths():
    for input_file in sorted(INPUT_TEST_DATA.glob("*.csv"), key=os.path.getsize):
        test_name = input_file.stem
        output_file = INPUT_TEST_DATA.parent / "output" / f"{test_name}_output.csv"

        yield input_file.as_posix(), output_file.as_posix()    

@pytest.fixture
def spark(scope="session"):
    spark = SparkSession.builder.appName("pyspark unittest").getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def df_schema():
    yield StructType([
        StructField("edge_1", IntegerType(), True),
        StructField("edge_2", IntegerType(), True),
        StructField("length", IntegerType(), True)
    ])

@pytest.fixture()
def example1_io(spark, df_schema):
    _input = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
    ], df_schema)

    _output = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (1, 3, 2),
        (1, 4, 3),
        (2, 4, 2)
    ], df_schema)

    yield _input, _output

@pytest.fixture()
def example2_io(spark, df_schema):
    _input = spark.createDataFrame([
        (1, 2, 1),
        (1, 3, 3),
        (2, 3, 1)
    ], df_schema)

    _output = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (1, 3, 2)
    ], df_schema)

    yield _input, _output

@pytest.fixture()
def cycle_io(spark, df_schema):
    _input = spark.createDataFrame([
        (1, 2, 1),
        (2, 3, 1),
        (3, 1, 1)
    ], df_schema)

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
    ], df_schema)

    yield _input, _output

@pytest.fixture()
def branch_io(spark, df_schema):
    _input = spark.createDataFrame([
        (1, 2, 1),
        (2, 4, 1),
        (2, 5, 1)
    ], df_schema)

    _output = spark.createDataFrame([
        (1, 2, 1),
        (2, 4, 1),
        (2, 5, 1),
        (1, 4, 2),
        (1, 5, 2),
    ], df_schema)

    yield _input, _output


@pytest.mark.parametrize("input_expected", [
    (pytest.lazy_fixture('example1_io')),
    (pytest.lazy_fixture('example2_io')),
    (pytest.lazy_fixture('cycle_io')),
])
def test_basic_io(input_expected):
    _input, expected = input_expected
    output = linear_paths_algorithm(_input)
    assertDataFrameEqual(output, expected)

#@pytest.mark.parametrize("input_path, output_path", list(get_test_data_paths()))
#def test_kj_io(spark, df_schema, input_path, output_path):
#    _input = spark.read.csv(input_path, header=True, schema=df_schema)
#    expected = spark.read.csv(output_path, header=True, schema=df_schema)
#    output = linear_paths_algorithm(_input)
#    assertDataFrameEqual(output, expected)
