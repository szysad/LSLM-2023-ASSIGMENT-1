import sys
import argparse
from pyspark.sql import SparkSession
from lib.linear_paths import linear_paths_algorithm
from lib.doubling_paths import doubling_paths_algorithm


def main():
    parser = argparse.ArgumentParser(description="Compute Path relation using Spark SQL algorithms.")
    parser.add_argument("algorithm", choices=["linear", "doubling"], help="Type of algorithm to use")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to the output CSV file")

    try:
        args = parser.parse_args()

        spark = SparkSession.builder.appName("PathRelation").getOrCreate()

        # Read input CSV file
        edges = spark.read.csv(args.input_file, header=True, inferSchema=True)

        # Apply the specified algorithm
        if args.algorithm == "linear":
            paths = linear_paths_algorithm(spark, edges)
        elif args.algorithm == "doubling":
            paths = doubling_paths_algorithm(spark, edges)
        else:
            raise ValueError("Invalid algorithm. Please choose 'linear' or 'doubling'")

        # Write the computed Path relation to the output CSV file
        paths.toPandas().to_csv(args.output_file, index=False)

        # Stop the Spark session
        spark.stop()
    except Exception as e:
        print(e)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
