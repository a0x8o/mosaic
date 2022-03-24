import py4j.java_gateway
from pyspark.sql import DataFrame, SparkSession

from mosaic.config import config


class DisplayHandler:
    MosaicFrameClass: py4j.java_gateway.JavaClass
    MosaicFrameObject: py4j.java_gateway.JavaObject
    ScalaOptionClass: py4j.java_gateway.JavaClass
    ScalaOptionObject: py4j.java_gateway.JavaObject
    in_databricks: bool
    display_function = None

    def __init__(self, spark: SparkSession):
        try:
            from PythonShellImpl import PythonShell

            self.display_function = PythonShell.display
            self.in_databricks = True
        except ImportError:
            self.display_function = self.basic_display
            self.in_databricks = False
        sc = spark.sparkContext
        self.ScalaOptionClass = getattr(sc._jvm.scala, "Option$")
        self.ScalaOptionObject = getattr(self.ScalaOptionClass, "MODULE$")
        self.PrettifierModule = getattr(
            sc._jvm.com.databricks.labs.mosaic.sql, "Prettifier"
        )

    @staticmethod
    def basic_display(df: DataFrame):
        df.show()

    def display(self, df: DataFrame):
        prettifier = self.PrettifierModule
        pretty_jdf = (
            prettifier.prettified(df._jdf, self.ScalaOptionObject.apply(None))
            if not type(df) == "MosaicFrame"
            else prettifier.prettifiedMosaicFrame(df._mosaicFrame)
        )
        pretty_df = DataFrame(pretty_jdf, config.sql_context)
        self.display_function(pretty_df)


def displayMosaic(df: DataFrame):
    if not hasattr(config, "display_handler"):
        config.display_handler = DisplayHandler(config.mosaic_spark)
    config.display_handler.display(df)