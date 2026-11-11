"""The occurrence order shared by hashing and GLM input generation."""

import pandas as pd


def ordered_occurrences(group):
    group = group.copy()
    group['start'] = pd.to_numeric(group['start'], errors='raise')
    if group['start'].isna().any() or not group['strand'].isin(['+', '-']).all():
        raise ValueError('Neighborhood occurrences require numeric starts and +/- strands')
    if not group['rep'].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError('Neighborhood occurrences require nonempty representatives')
    return group.sort_values('start', kind='stable').drop_duplicates('start').reset_index(drop=True)


def query_labels(values):
    """Normalize scalar/list query labels without losing multiple search hits."""
    result = []
    for value in values:
        for query in value if isinstance(value, (list, tuple)) else [value]:
            if pd.isna(query):
                continue
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f'Invalid query label: {query!r}')
            if query not in result:
                result.append(query)
    return result
