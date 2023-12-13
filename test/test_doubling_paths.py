import pytest
from pyspark.testing.utils import assertDataFrameEqual
from lib.doubling_paths import doubling_paths_algorithm
from test.conftest import get_test_data_paths


@pytest.mark.parametrize("input_expected", [
    (pytest.lazy_fixture('example1_io')),
    (pytest.lazy_fixture('example2_io')),
    (pytest.lazy_fixture('cycle_io')),
    (pytest.lazy_fixture('shortcut_io')),
    (pytest.lazy_fixture('redundand_edge')),
])
def test_basic_io(spark, input_expected):
    _input, expected = input_expected
    output = doubling_paths_algorithm(spark, _input)
    assertDataFrameEqual(output, expected)

@pytest.mark.parametrize("input_path, output_path", list(get_test_data_paths()))
def test_kj_io(spark, edge_schema, input_path, output_path):
    _input = spark.read.csv(input_path, header=True, schema=edge_schema)
    expected = spark.read.csv(output_path, header=True, schema=edge_schema)
    output = doubling_paths_algorithm(spark, _input)
    assertDataFrameEqual(output, expected)

def test_20_100_subset(spark, kj_test_20_100):
    _input, output_subset = kj_test_20_100
    output = doubling_paths_algorithm(spark, _input)

    intersection = output_subset.intersect(output)

    assertDataFrameEqual(intersection, output_subset)

@pytest.mark.parametrize("input_expected_subset", [
    (pytest.lazy_fixture('kj_test_20_100')),
    (pytest.lazy_fixture('partial_shortcut')),
    (pytest.lazy_fixture('only_length_updates')),
    (pytest.lazy_fixture('only_length_updates_2')),
])
def test_subset_inclusion(spark, input_expected_subset):
    _input, output_subset = input_expected_subset
    output = doubling_paths_algorithm(spark, _input)

    intersection = output_subset.intersect(output)

    assertDataFrameEqual(intersection, output_subset)
