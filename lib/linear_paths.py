from pyspark.sql import functions as F
import pyspark.sql as sql
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType
from lib.utils import serialize_and_deserialize


path_processing_schema = StructType([
        StructField("edge_1", IntegerType(), False),
        StructField("edge_2", IntegerType(), False),
        StructField("length", IntegerType(), False),
        StructField("expanded", BooleanType(), False)
    ])


def linear_paths_algorithm(spark: sql.SparkSession, edges: sql.DataFrame) -> sql.DataFrame:
    # Initialize the path DataFrame with single-edge paths
    edges = (edges
        .alias("edges")
        .cache()
    )
    paths = (edges
        .alias("paths")
        .withColumn("expanded", F.lit(False))
        .cache()
    )

    while True:
        extended_paths = (paths
            .filter(paths.expanded == False)
            .join(edges, F.col("paths.edge_2") == F.col("edges.edge_1"), how="inner")
            .selectExpr("paths.edge_1", "edges.edge_2", "paths.length + edges.length as length", "False as expanded")
        )

        # these path might have duplicates with bigger length paths
        paths = (paths
            .unpersist()
            .withColumn("expanded", F.lit(True))
            .union(extended_paths)
            .groupby(paths.edge_1, paths.edge_2)
            .agg(F.expr("min(length) as length"), F.expr("min_by(expanded, length) as expanded"))
        )

        # speeds up execution
        # https://stackoverflow.com/questions/73856043/pyspark-loop-n-times-each-loop-gets-gradually-slower/73865157#73865157
        paths = serialize_and_deserialize(spark, path_processing_schema, paths) \
            .alias("paths")

        paths.cache()

        # finish if number of expanded paths is lq 1
        if paths.filter(paths.expanded == False).count() <= 1:
            break


    edges.unpersist()
    paths.unpersist()

    return paths.select(paths.edge_1, paths.edge_2, paths.length)
