import pandas as pd
import numpy as np


# 因子分析
def corr_selection(df, score_col):
    df_pair = (
        df[df.index]
        .unstack()
        .drop_duplicates()
        .where(lambda x: (abs(x) >= 0.70) & (abs(x) != 1))
        .dropna()
        .reset_index()
    )
    df_pair.columns = ["factor_1", "factor_2", "corr"]
    df_pair["score_1"] = df[score_col][df_pair["factor_1"]].to_list()
    df_pair["score_2"] = df[score_col][df_pair["factor_2"]].to_list()
    df_pair["drop_factor"] = df_pair["factor_1"].mask(
        df_pair["score_1"] < df_pair["score_2"], df_pair["factor_2"]
    )
    df_pair["drop_factor"] = df_pair["drop_factor"].mask(
        df_pair["score_1"] == df_pair["score_2"], np.nan
    )
    return df_pair


def factor_analysis(df, factor_dict, target_cols, date_col, return_col):
    require_cols = [x for x in factor_dict.values()]
    require_cols = [x for y in require_cols for x in y]
    # IC analysis
    df_IC = df.groupby(date_col).apply(
        lambda x: x[[target_cols] + require_cols].corr(method="spearman").iloc[0, 1:]
    )
    df_IC_mean = df_IC.mean()
    df_IC_std = df_IC.std()
    df_IR = df_IC_mean / df_IC_std
    df_result = pd.DataFrame()
    df_result["IC_mean"] = df_IC_mean
    df_result["IC_std"] = df_IC_std
    df_result["IR"] = df_IR
    df_result["IR_abs"] = df_IR.abs()
    group_list = [[key] * len(value) for key, value in factor_dict.items()]
    group_list = [x for y in group_list for x in y]
    df_result["factor_group"] = group_list
    df_result["min_nunique"] = df.groupby(date_col)[require_cols].nunique().min()
    df_result.index.name = "factor"
    # 5 claw portfolio analysis
    lst_group_mean_return_prod_all = []
    for factor, min_nunique in zip(df_result.index, df_result["min_nunique"]):
        if min_nunique > 5:
            df[factor + "_group"] = df.groupby(date_col, group_keys=False)[
                factor
            ].apply(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
            df_group_mean_return = (
                df.groupby([date_col, factor + "_group"], as_index=False)[return_col]
                .mean()
                .pivot(index=date_col, columns=factor + "_group", values=return_col)
            )
            lst_group_mean_return_prod = (df_group_mean_return + 1).prod().to_list()
            top_bottom = abs(
                lst_group_mean_return_prod[-1] - lst_group_mean_return_prod[0]
            )
            lst_group_mean_return_prod += [top_bottom]
        elif min_nunique > 0 and min_nunique <= 5:
            df[factor + "_group"] = df.groupby(date_col, group_keys=False)[
                factor
            ].apply(lambda x: pd.qcut(x, min_nunique, labels=False, duplicates="drop"))

            df_group_mean_return = (
                df.groupby([date_col, factor + "_group"], as_index=False)[return_col]
                .mean()
                .pivot(index=date_col, columns=factor + "_group", values=return_col)
            )
            lst_group_mean_return_prod = (df_group_mean_return + 1).prod().to_list()
            top_bottom = abs(
                lst_group_mean_return_prod[-1] - lst_group_mean_return_prod[0]
            )
            lst_group_mean_return_prod += [np.NaN] * (
                5 - len(lst_group_mean_return_prod)
            )
            lst_group_mean_return_prod += [top_bottom]
        else:
            lst_group_mean_return_prod = [np.NaN] * 6
        lst_group_mean_return_prod_all.append(lst_group_mean_return_prod)
    df_result[["1", "2", "3", "4", "5", "top-bottom"]] = pd.DataFrame(
        lst_group_mean_return_prod_all, index=df_result.index
    )
    # corr analysis
    df_corr = (
        df.groupby(date_col)
        .apply(lambda x: x[require_cols].corr())
        .groupby(level=1)
        .mean()
    )
    df_corr = df_corr.reindex(df_result.index)
    df_result[df_corr.columns] = df_corr
    # rank
    df_result["top-bottom_rank"] = df_result.groupby("factor_group")["top-bottom"].rank(
        ascending=False
    )
    df_result["IR_rank"] = df_result.groupby("factor_group")["IR_abs"].rank(
        ascending=False
    )
    df_result["score_rank"] = df_result["top-bottom_rank"] + df_result["IR_rank"]
    org_index = df_result.index
    df_result = df_result.groupby(
        "factor_group", as_index=False, group_keys=False
    ).apply(lambda x: x.sort_values(["score_rank", "top-bottom_rank", "IR_rank"]))
    df_result["score"] = df_result.groupby(
        "factor_group", as_index=False, group_keys=False
    )["score_rank"].apply(lambda x: pd.Series(np.arange(len(x)), index=x.index))
    df_drop = (
        df_result.groupby("factor_group")
        .apply(lambda x: corr_selection(x, "score"))
        .reset_index()
    )
    df_result["corr_select"] = df_result.index.isin(df_drop["drop_factor"]) == False
    df_result_select = (
        df_result[df_result["corr_select"]]
        .groupby("factor_group", as_index=False, group_keys=False)
        .apply(
            lambda x: x.sort_values("score").head(
                int(len(x) * 0.5) + 1 if len(x) % 2 != 0 else int(len(x) * 0.5)
            )
        )
        .reset_index()
    )
    return df_result.reindex(org_index), df_drop, df_result_select
