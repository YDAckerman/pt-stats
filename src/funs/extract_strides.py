from whittaker_eilers import WhittakerSmoother
import numpy as np


def extract_strides(df, var_name, start=0, end=-1):

    whittaker_smoother = WhittakerSmoother(
        lmbda=100, order=2, data_length=df.shape[0]
    )

    df['whit_smooth'] = whittaker_smoother.smooth(df[var_name].values)

    peaks = [i for i, v in enumerate(df.whit_smooth)
             if (i > 0 and i < df.shape[0] - 1) and
             v > df.whit_smooth.values[i-1] and
             v > df.whit_smooth.values[i+1] and
             v > np.quantile(df.whit_smooth.values, .9)]

    troughs = [i for i, v in enumerate(df.whit_smooth)
               if (i > 0 and i < df.shape[0] - 1) and
               v < df.whit_smooth.values[i-1] and
               v < df.whit_smooth.values[i+1] and
               i > peaks[0] and i < peaks[-1]]

    if end > len(troughs) - 1:
        end = -1

    return df.loc[troughs[start]:troughs[end]].copy()
