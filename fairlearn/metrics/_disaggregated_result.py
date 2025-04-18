# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

import logging
from itertools import product
from typing import Literal

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoDataFrame, IntoSeries

from ._annotated_metric_function import AnnotatedMetricFunction

logger = logging.getLogger(__name__)

_VALID_ERROR_STRING = ["raise", "coerce"]
_VALID_GROUPING_FUNCTION = ["min", "max"]

_INVALID_ERRORS_VALUE_ERROR_MESSAGE = "Invalid error value specified. Valid values are {0}".format(
    _VALID_ERROR_STRING
)
_INVALID_GROUPING_FUNCTION_ERROR_MESSAGE = (
    "Invalid grouping function specified. Valid values are {0}".format(_VALID_GROUPING_FUNCTION)
)
_MF_CONTAINS_NON_SCALAR_ERROR_MESSAGE = (
    "Metric frame contains non-scalar cells. Please remove non-scalar columns from your"
    " metric frame or use parameter errors='coerce'."
)


def apply_to_dataframe(
    data: IntoDataFrame,
    metric_functions: dict[str, AnnotatedMetricFunction],
    include_groups: bool = False,
) -> np.ndarray:
    """Apply metric functions to a DataFrame.

    The incoming DataFrame may have been sliced via `groupby()`.
    This function applies each annotated function in turn to the
    supplied DataFrame.

    The include_groups argument is weird. It appears that pandas
    introduced it as an argument in v2.2, and immediately deprecated
    it (dependent on when this is being read, may need to adjust):
    https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.apply.html
    We don't use this argument, and only include it so that we can be
    compatible with pandas<2.2
    """
    return np.array([metric_function(data) for metric_function in metric_functions.values()])


class DisaggregatedResult:
    """Pickier version of MetricFrame.

    This holds the internal result from a disaggregated metric
    computation, and provides `apply_grouping()` (to cover min
    and max), `difference()` and `ratio()` methods.

    The main difference to the results computed by MetricFrame
    is that no account is made of whether the user supplied
    a bare function or a dictionary. Hence the results are
    always Series or DataFrame.

    Parameters
    ----------
    overall: Series or DataFrame
        The metric function(s) computed on the entire dataset, split by
        control features if supplied
    by_group: Series or DataFrame
        The metric function(s) computed on each subgroup identified by
        the sensitive and control features
    """

    def __init__(self, overall: IntoSeries | IntoDataFrame, by_group: IntoDataFrame):
        """Construct an object."""
        self._overall = nw.from_native(
            overall, eager_only=True, allow_series=True, pass_through=False
        )
        self._by_group = nw.from_native(by_group, eager_only=True, pass_through=False)

    @property
    def overall(self) -> IntoSeries | IntoDataFrame:
        """Return overall metrics."""
        return self._overall.to_native()

    @property
    def by_group(self) -> IntoDataFrame:
        """Return the metrics by group."""
        return self._by_group.to_native()

    def apply_grouping(
        self,
        grouping_function: Literal["min", "max"],
        control_feature_names: list[str] | None = None,
        errors: Literal["raise", "coerce"] = "raise",
    ) -> IntoSeries | IntoDataFrame:
        """Compute mins or maxes.

        Parameters
        ----------
        grouping_function: string {'min', 'max'}
        control_feature_names: list[str] | None
            Names of the control features. Must appear in the index of the `overall`
            and `by_group` properties
        errors: string {'raise', 'coerce'}, default :code:`raise`
            How to deal with any errors. Either coerce to `np.nan` or wrap the
            exception and reraise

        Returns
        -------
        Series or DataFrame
            Contains the desired mins or maxes
        """
        if grouping_function not in _VALID_GROUPING_FUNCTION:
            raise ValueError(_INVALID_GROUPING_FUNCTION_ERROR_MESSAGE)

        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        if not control_feature_names:
            if errors == "raise":
                try:
                    result = self.by_group.agg(grouping_function, axis=0)
                except ValueError as ve:
                    raise ValueError(_MF_CONTAINS_NON_SCALAR_ERROR_MESSAGE) from ve

            elif errors == "coerce":
                # Fill in the possible min/max values, else np.nan
                mf = self.by_group.apply(
                    lambda x: x.apply(lambda y: y if np.isscalar(y) else np.nan)
                )
                result = mf.agg(grouping_function, axis=0)
        else:
            if errors == "raise":
                try:
                    result = self.by_group.groupby(level=control_feature_names).agg(
                        grouping_function
                    )

                except ValueError as ve:
                    raise ValueError(_MF_CONTAINS_NON_SCALAR_ERROR_MESSAGE) from ve
            elif errors == "coerce":
                # Fill all impossible columns with NaN before grouping metric frame
                mf = self.by_group.apply(
                    lambda x: x.apply(lambda y: y if np.isscalar(y) else np.nan)
                )
                result = mf.groupby(level=control_feature_names).agg(grouping_function)

        assert isinstance(result, pd.Series) or isinstance(result, pd.DataFrame)

        return result

    def difference(
        self,
        control_feature_names: list[str] | None = None,
        method: Literal["between_groups", "to_overall"] = "between_groups",
        errors: Literal["raise", "coerce"] = "coerce",
    ) -> IntoSeries | IntoDataFrame:
        """Return the maximum absolute difference between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the maximum absolute difference between the entries in each
        combination of sensitive features in the :attr:`.by_group` property.

        There are two allowed values for the ``method=`` parameter. The
        value ``between_groups`` computes the maximum difference between
        any two pairs of groups in the :attr:`.by_group` property (i.e.
        ``group_max() - group_min()``). Alternatively, ``to_overall``
        computes the difference between each subgroup and the
        corresponding value from :attr:`.overall` (if there are control
        features, then :attr:`.overall` is multivalued for each metric).
        The result is the absolute maximum of these values.

        Parameters
        ----------
        control_feature_names: list[str] | None
            Names of the control features. Must appear in the index of the `overall`
            and `by_group` properties
        method : {'between_groups', 'overall'}, default :code:`between_groups`
            How to compute the aggregate.
        errors: {'raise', 'coerce'}, default :code:`coerce`
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        pandas.Series or pandas.DataFrame
        """
        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        if method == "between_groups":
            subtrahend = self.apply_grouping("min", control_feature_names, errors=errors)
        elif method == "to_overall":
            subtrahend = self.overall
        else:
            raise ValueError("Unrecognised method '{0}' in difference() call".format(method))

        # Can assume errors='coerce', else error would already have been raised in .group_min
        # Fill all non-scalar values with NaN
        mf = self.by_group.apply(lambda x: x.apply(lambda y: y if np.isscalar(y) else np.nan))

        if control_feature_names is None:
            result = (mf - subtrahend).abs().max()
        else:
            result = (mf - subtrahend).abs().groupby(level=control_feature_names).max()

        assert isinstance(result, pd.Series) or isinstance(result, pd.DataFrame)

        return result

    def ratio(
        self,
        control_feature_names: list[str] | None = None,
        method: Literal["between_groups", "to_overall"] = "between_groups",
        errors: Literal["raise", "coerce"] = "coerce",
    ) -> IntoSeries | IntoDataFrame:
        """Return the minimum ratio between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the minimum ratio (that is, the ratio is forced to be
        less than unity) between the entries in each
        column of the :attr:`.by_group` property.

        There are two allowed values for the ``method=`` parameter. The
        value ``between_groups`` computes the minimum ratio between
        any two pairs of groups in the :attr:`.by_group` property (i.e.
        ``group_min() / group_max()``). Alternatively, ``to_overall``
        computes the ratio between each subgroup and the
        corresponding value from :attr:`.overall` (if there are control
        features, then :attr:`.overall` is multivalued for each metric),
        expressing the ratio as a number less than 1.
        The result is the minimum of these values.

        Parameters
        ----------
        control_feature_names: list[str] | None
            Names of the control features. Must appear in the index of the `overall`
            and `by_group` properties
        method : {'between_groups', 'overall'}, default :code:`between_groups`
            How to compute the aggregate.
        errors: {'raise', 'coerce'}, default :code:`coerce`
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
        """

        def ratio_sub_one(x):
            if x > 1:
                return 1 / x
            else:
                return x

        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        if method == "between_groups":
            result = self.apply_grouping(
                "min", control_feature_names, errors=errors
            ) / self.apply_grouping("max", control_feature_names, errors=errors)
        elif method == "to_overall":
            ratios = None

            if control_feature_names is not None:
                # It's easiest to give in to the DataFrame columns preference
                ratios = self.by_group.unstack(level=control_feature_names) / self.overall.unstack(
                    level=control_feature_names
                )
            else:
                ratios = self.by_group / self.overall

            ratios = ratios.apply(lambda x: x.transform(ratio_sub_one))
            if not control_feature_names:
                result = ratios.min()
            else:
                result = ratios.min().unstack(0)
        else:
            raise ValueError("Unrecognised method '{0}' in ratio() call".format(method))

        assert isinstance(result, pd.Series) or isinstance(result, pd.DataFrame)

        return result

    @staticmethod
    def create(
        *,
        data: IntoDataFrame,
        annotated_functions: dict[str, AnnotatedMetricFunction],
        sensitive_feature_names: list[str],
        control_feature_names: list[str] | None = None,
    ) -> "DisaggregatedResult":
        """Manufacture a DisaggregatedResult.

        This is essentially a more restricted version of the MetricFrame
        constructor.

        All of the `data` have to be supplied as a DataFrame.
        The metric functions have to be supplied as a dictionary of
        AnnotatedMetricFunction.
        The latter class contains the metric function itself, and
        mappings between the metric function arguments and the columns
        of the `data` DataFrame.
        The sensitive and (optional) control features are lists of
        column names in `data`.

        Parameters
        ----------
        data : DataFrame
            A DataFrame containing all of the columns required to compute the metrics
        annotated_functions: dict[str, AnnotatedMetricFunction]
            A dictionary of metric functions, each of which is annotated with the
            mapping of columns in `data` to argument names in the function
        sensitive_feature_names: list[str]
            The list of columns in `data` which correspond to the sensitive feature(s)
        control_feature_names: list[str] | None
            Optional list of columns in `data` which correspond to the control features,
            if any

        Returns
        -------
        DisaggregatedResult
            Freshly constructed instance of this class
        """
        overall = DisaggregatedResult._apply_functions(
            data=data,
            annotated_functions=annotated_functions,
            grouping_names=control_feature_names,
        )

        by_group = DisaggregatedResult._apply_functions(
            data=data,
            annotated_functions=annotated_functions,
            grouping_names=(control_feature_names or []) + sensitive_feature_names,
        )

        return DisaggregatedResult(overall, by_group)

    @staticmethod
    def _apply_functions(
        *,
        data: IntoDataFrame,
        annotated_functions: dict[str, AnnotatedMetricFunction],
        grouping_names: list[str] | None,
    ) -> IntoSeries | IntoDataFrame:
        """
        Apply annotated metric functions to a DataFrame, optionally grouping by specified columns.

        Parameters
        ----------
        data : pd.DataFrame
            The input data on which the metric functions will be applied.
        annotated_functions : dict[str, AnnotatedMetricFunction]
            A dictionary where keys are metric names and values are the corresponding annotated metric
            functions.
        grouping_names : list[str] | None
            A list of column names to group by before applying the metric functions. If None, the
            functions are applied to the entire DataFrame.

        Returns
        -------
        Series or DataFrame
            A Series or DataFrame with the results of the metric functions applied. If grouping_names is provided,
            the results are grouped accordingly.
        """

        names = list(annotated_functions.keys())

        nw_data = nw.from_native(data, eager_only=True)
        implementation = nw_data.implementation

        if grouping_names is None or len(grouping_names) == 0:
            output_data = {
                "metric": names,
                "score": apply_to_dataframe(data, metric_functions=annotated_functions),
            }
            result = nw.maybe_set_index(
                nw.from_dict(
                    output_data,
                    schema={"metric": nw.String(), "score": nw.Float64()},
                    backend=implementation,
                ),
                "metric",
            ).to_native()

            return (
                result["score"].rename(None).rename_axis(None)
                if implementation.is_pandas_like()
                else result
            )

        input_schema = nw_data.schema
        group_names_schema = {
            group_name: input_schema[group_name] for group_name in grouping_names
        }
        output_schema = {**group_names_schema, **{name: nw.Float64() for name in names}}
        temp = np.vstack(
            [
                np.hstack(
                    [
                        *key,
                        apply_to_dataframe(
                            data=group, metric_functions=annotated_functions, include_groups=False
                        ),
                    ]
                )
                for key, group in nw_data.group_by(grouping_names)
            ]
        )

        result = nw.from_numpy(temp, schema=output_schema, backend=implementation)

        if len(grouping_names) > 1:
            _tmp = nw.from_numpy(
                data=np.stack([*product(*[np.unique(nw_data[col]) for col in grouping_names])]),
                schema=group_names_schema,
                backend=implementation,
            )
            result = _tmp.join(result, on=grouping_names, how="left")

        return nw.maybe_set_index(result, grouping_names).to_native()
