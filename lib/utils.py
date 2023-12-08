from pyspark.sql import functions as F


def linear_paths_algorithm(edges):
    # Initialize the path DataFrame with single-edge paths
    edges = edges.alias("edges").cache()
    paths = edges.alias("paths").withColumn("expanded", F.lit(False))
    paths.cache()
    paths_cnt_after = paths.count()

    while True:
        paths_cnt_before = paths_cnt_after

        extended_paths = (paths
            .filter(paths.expanded == False)
            .join(edges, F.col("paths.edge_2") == F.col("edges.edge_1"), how="inner")
            .selectExpr("paths.edge_1", "edges.edge_2", "paths.length + edges.length as length", "False as expanded")
        )

        paths.unpersist()

        # these path might have duplicates with bigger length paths
        paths = (paths  
            .withColumn("expanded", F.lit(True))
            .union(extended_paths)
            .groupby(paths.edge_1, paths.edge_2)
            .agg(F.expr("min(length) as length"), F.expr("first(expanded) as expanded"))
            .alias("paths")
        )

        paths.cache()

        paths_cnt_after = paths.count()

        # stop if no new paths were added
        if paths_cnt_before == paths_cnt_after:
            break

    edges.unpersist()
    paths.unpersist()

    return paths.select(paths.edge_1, paths.edge_2, paths.length)
