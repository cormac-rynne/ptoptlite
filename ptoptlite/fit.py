# fit.py
import numpy as np
import pandas as pd
from pyspark.sql import functions as f
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import DoubleType, MapType, StringType, StructField, StructType
from scipy.optimize import curve_fit  # Import inside the function for serialization

# Define the schema outside the function as it doesn't depend on dynamic parameters
schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("params", MapType(StringType(), DoubleType()), False),
        StructField("r_squared", DoubleType(), True),
    ]
)


def fit_curves(df, id_col, x_col, y_col, bounds, curve_function, param_names):
    """
    Fits a curve to each group in the DataFrame and returns the fitted parameters and R-squared values.

    Parameters:
    - df: Spark DataFrame containing the data.
    - id_col: Column name to group by.
    - x_col: Column name for the x-axis data.
    - y_col: Column name for the y-axis data.
    - bounds: Bounds for the curve fitting parameters.
    - curve_function: The function to fit to the data.
    - param_names: List of parameter names for the curve function.

    Returns:
    - Spark DataFrame with fitted parameters and R-squared values for each group.
    """

    # Define the Pandas UDF inside the fit_curves function to capture the parameters
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def fit_curve_udf(pdf):
        # Extract data
        xdata = pdf[x_col].values
        ydata = pdf[y_col].values

        # Initialize default values
        params_dict = {}  # Initialize as empty dict
        r_squared = np.nan

        # Ensure there are enough data points
        if len(xdata) >= len(param_names):  # Typically, you need at least as many points as parameters
            try:
                # Perform curve fitting
                popt, pcov = curve_fit(curve_function, xdata, ydata, bounds=bounds)
                params_dict = {name: float(param) for name, param in zip(param_names, popt)}

                # Calculate residuals and R-squared
                residuals = ydata - curve_function(xdata, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            except Exception as e:
                # Optionally, log the exception if logging is set up
                # logger.warning(f"Curve fitting failed for id {pdf[id_col].iloc[0]}: {e}")
                pass  # params_dict remains empty and r_squared remains NaN

        return pd.DataFrame({"id": [pdf[id_col].iloc[0]], "params": [params_dict], "r_squared": [r_squared]})

    # Apply the UDF to each group
    model_params = df.groupBy(id_col).apply(fit_curve_udf)

    # Extract individual parameters from the 'params' map
    for param in param_names:
        model_params = model_params.withColumn(param, f.col("params").getItem(param).cast(DoubleType()))

    # Optionally, drop the 'params' column if it's no longer needed
    model_params = model_params.drop("params")

    return model_params
