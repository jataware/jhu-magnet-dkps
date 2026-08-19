"""
Carry a (dataset, metric) pair through a matrix as one value.

The cards sweep 11 paired configurations. A kwdagger matrix crosses its axes,
so sweeping dataset and metric separately would produce combinations the cards
never asked for. One axis of ``dataset|metric`` strings keeps the pairing.

The separator is ``|`` because dataset names contain colons, e.g.
``legalbench:subset=abercrombie``.
"""


def split_dataset_cfg(value):
    """
    Args:
        value (str): ``"<dataset>|<metric>"``.

    Returns:
        tuple: ``(dataset, metric)``

    Example:
        >>> from jhu_ta1.magnet._dataset_cfg import split_dataset_cfg
        >>> split_dataset_cfg('med_qa|quasi_exact_match')
        ('med_qa', 'quasi_exact_match')
        >>> split_dataset_cfg('legalbench:subset=proa|quasi_exact_match')
        ('legalbench:subset=proa', 'quasi_exact_match')
    """
    dataset, sep, metric = value.rpartition('|')
    if not sep:
        raise ValueError(
            f'dataset_cfg {value!r} needs the form "<dataset>|<metric>"')
    return dataset, metric
