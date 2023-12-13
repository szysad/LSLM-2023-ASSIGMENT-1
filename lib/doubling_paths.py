from pyspark.sql import functions as F
import pyspark.sql as sql
from pyspark.sql.types import StructType, StructField, IntegerType
from lib.utils import serialize_and_deserialize


path_processing_schema = StructType([
        StructField("edge_1", IntegerType(), False),
        StructField("edge_2", IntegerType(), False),
        StructField("length", IntegerType(), False),
        StructField("epoch", IntegerType(), False)
    ])


def doubling_paths_algorithm(spark: sql.SparkSession, edges: sql.DataFrame) -> sql.DataFrame:
    epoch = 0
    max_path_epoch = 0
    paths = (edges
        .alias("paths")
        .withColumn("epoch", F.lit(epoch))
    )

    paths.cache()
    # iterate untill no new edges are created
    while epoch == max_path_epoch:
        new_paths = paths.filter(paths.epoch == epoch).alias("new_paths")
        old_paths = paths.filter(paths.epoch != epoch).alias("old_paths")

        epoch += 1

        paths.unpersist()
        new_paths.cache()
        old_paths.cache()

        # all new possible paths between new paths
        new_paths_b1 = (new_paths.alias("p1")
            .join(new_paths.alias("p2"), F.col("p1.edge_2") == F.col("p2.edge_1"), how="inner")
            .selectExpr("p1.edge_1", "p2.edge_2", "p1.length + p2.length as length")
            .withColumn("epoch", F.lit(epoch))
        )

        # all new possible paths between new paths and old paths
        new_paths_b2 = (new_paths
            .join(old_paths, F.col("new_paths.edge_2") == F.col("old_paths.edge_1"), how="inner")
            .selectExpr("new_paths.edge_1", "old_paths.edge_2", "new_paths.length + old_paths.length as length")
            .withColumn("epoch", F.lit(epoch))
        )

        paths = (old_paths
            .union(new_paths)
            .union(new_paths_b1)
            .union(new_paths_b2)
            .select("edge_1", "edge_2", "length", F.struct("length", "epoch").alias("length_epoch"))
            .groupby("edge_1", "edge_2")
            .agg(
                F.expr("min(length) as length"),
                F.expr("min_by(length_epoch.epoch, length_epoch) as epoch") # get the epoch corresponding to the row with min (length, epoch)
            )
        )

        old_paths.unpersist()
        new_paths.unpersist()

        # speeds up execution
        # https://stackoverflow.com/questions/73856043/pyspark-loop-n-times-each-loop-gets-gradually-slower/73865157#73865157
        paths = serialize_and_deserialize(spark, path_processing_schema, paths) \
            .alias("paths")

        paths.cache()
        max_path_epoch = paths.agg(F.max(paths.epoch)).collect()[0][0]


    paths.unpersist()

    return paths.select(paths.edge_1, paths.edge_2, paths.length)
