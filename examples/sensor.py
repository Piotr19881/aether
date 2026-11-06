from collections.abc import Sequence


def normalize_voltage(samples: Sequence[Sequence[float]]) -> list[list[float]]:
    values = [value for row in samples for value in row]
    mean_voltage = sum(values) / len(values)
    centered = [[value - mean_voltage for value in row] for row in samples]
    maximum = max(abs(value) for row in centered for value in row)
    return [[value / maximum for value in row] for row in centered]
