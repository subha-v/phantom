import pandas as pd
import numpy as np

df = pd.read_csv("OpenBCI-RAW-2025-09-13_23-58-04.csv", skiprows=4)
print(f"Total samples: {len(df)}")
print(f"Columns with 'Marker': {[col for col in df.columns if 'Marker' in col]}")

marker_col = " Marker"
if marker_col in df.columns:
    markers = df[marker_col]
    print(f"Non-zero markers: {(markers != 0).sum()}")
    print(f"Unique marker values: {markers.unique()[:10]}")

    first_nonzero = markers[markers != 0].index[0] if (markers != 0).any() else None
    if first_nonzero:
        print(f"First non-zero marker at index: {first_nonzero}")