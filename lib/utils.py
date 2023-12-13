from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import StructType


def serialize_and_deserialize(spark: SparkSession, schema: StructType, df: sql.DataFrame) -> sql.DataFrame:
    return spark.createDataFrame(df.rdd, schema)
