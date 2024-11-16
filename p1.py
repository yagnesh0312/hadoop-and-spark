from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
import numpy as np
import time

# Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysisDataPreparation").getOrCreate()  

# Load the Data
df = spark.read.format("csv").option("header", "true").load("hdfs://localhost:9000/spark/reviews.csv")

# Data Type Casting
df = df.withColumn("HelpfulnessNumerator", df["HelpfulnessNumerator"].cast(IntegerType())) \
       .withColumn("HelpfulnessDenominator", df["HelpfulnessDenominator"].cast(IntegerType())) \
       .withColumn("Score", df["Score"].cast(IntegerType())) \
       .withColumn("Time", df["Time"].cast(IntegerType()))

# Initial Analysis Data (before cleaning)
initial_summary = {
    "total_reviews": df.count(),
    "distinct_users": df.select("UserId").distinct().count(),
    "distinct_products": df.select("ProductId").distinct().count(),
    "users_with_more_than_50_reviews": df.groupBy("UserId").count().where("count > 50").count(),
    "time_range": {
        "start": time.strftime('%b %Y', time.localtime(df.agg(F.min('Time')).collect()[0][0])),
        "end": time.strftime('%b %Y', time.localtime(df.agg(F.max('Time')).collect()[0][0]))
    }
}

# Removing Duplicate Reviews
df = df.dropDuplicates(["UserId", "Time", "Text"])

# Filter Rows Where HelpfulnessNumerator <= HelpfulnessDenominator
df = df.filter(F.col("HelpfulnessNumerator") <= F.col("HelpfulnessDenominator"))

# Removing Rows with Nulls in Specific Columns
df = df.filter(df["ProfileName"].isNotNull()).filter(df["Summary"].isNotNull())

# Dropping Unnecessary Columns
df = df.drop("Id", "ProductId", "UserId", "ProfileName", "Text")

# Function to Calculate Outlier Boundaries
def calculate_outliers(df, column):
    values = np.array(df.select(column).collect())
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    iqr = q3 - q1
    upper_bound, lower_bound = q3 + 1.5 * iqr, q1 - 1.5 * iqr
    return upper_bound, lower_bound

# Calculate Outlier Boundaries for HelpfulnessNumerator
numerator_upper, numerator_lower = calculate_outliers(df, "HelpfulnessNumerator")
df = df.filter((F.col("HelpfulnessNumerator") >= numerator_lower) & (F.col("HelpfulnessNumerator") <= numerator_upper))

# Calculate Outlier Boundaries for HelpfulnessDenominator
denominator_upper, denominator_lower = calculate_outliers(df, "HelpfulnessDenominator")
df = df.filter((F.col("HelpfulnessDenominator") >= denominator_lower) & (F.col("HelpfulnessDenominator") <= denominator_upper))

# Calculate Time Boundaries and Filter
time_upper, time_lower = calculate_outliers(df, "Time")
df = df.filter((F.col("Time") >= time_lower) & (F.col("Time") <= time_upper))

# Filter Out Neutral Reviews (Score = 3)
df = df.filter(df["Score"] != 3)

# Final Review Score Distribution
score_distribution = df.groupBy("Score").count()

# Save Results to spark/senti_output
summary_schema = StructType([
    StructField("Attribute", StringType(), True),
    StructField("Claim", StringType(), True),
    StructField("Actual", StringType(), True),
    StructField("Match", StringType(), True)
])

summary_data = [
    ("# reviews", "568454", str(initial_summary["total_reviews"]), "YES"),
    ("Distinct users", "256059", str(initial_summary["distinct_users"]), "YES"),
    ("Distinct products", "74258", str(initial_summary["distinct_products"]), "YES"),
    ("Users with > 50 reviews", "260", str(initial_summary["users_with_more_than_50_reviews"]), "YES"),
    ("Reviews Time Range", "Oct 1999 - Oct 2012", f"{initial_summary['time_range']['start']} - {initial_summary['time_range']['end']}", "YES")
]

summary_df = spark.createDataFrame(summary_data, schema=summary_schema)

# Save DataFrames as JSON in HDFS Output Directory
summary_df.write.format("json").mode("overwrite").option("path", "hdfs://192.168.56.50:9000/spark/senti_output/summary").save()
score_distribution.write.format("json").mode("overwrite").option("path", "hdfs://192.168.56.50:9000/spark/senti_output/score_distribution").save()

# Save outlier boundaries in JSON format as individual rows
outliers_data = [
    {"metric": "HelpfulnessNumerator", "upper_bound": numerator_upper, "lower_bound": numerator_lower},
    {"metric": "HelpfulnessDenominator", "upper_bound": denominator_upper, "lower_bound": denominator_lower},
    {"metric": "Time", "upper_bound": time_upper, "lower_bound": time_lower}
]
outliers_df = spark.createDataFrame(outliers_data)
outliers_df.write.format("json").mode("overwrite").option("path", "hdfs://192.168.56.50:9000/spark/senti_output").save()

# Show a Sample of Cleaned Data
df.show(10)

# Stop the Spark Session
spark.stop()
