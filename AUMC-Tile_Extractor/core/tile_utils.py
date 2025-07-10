def get_closest_level_for_mpp(reader, target_mpp: float) -> int:
    """
    Find the level in the WSI that has the closest MPP (microns per pixel) to the target.

    Parameters:
    - reader: An object implementing get_level_count() and get_mpp_for_level(level)
    - target_mpp (float): Desired MPP (e.g., 0.5 for 40x)

    Returns:
    - int: Closest matching level index
    """
    if not hasattr(reader, 'get_level_count') or not hasattr(reader, 'get_mpp_for_level'):
        raise AttributeError("Reader must implement get_level_count() and get_mpp_for_level(level)")

    min_diff = float('inf')
    closest_level = 0

    for level in range(reader.get_level_count()):
        level_mpp = reader.get_mpp_for_level(level)
        if level_mpp is None:
            continue  # skip invalid levels
        diff = abs(level_mpp - target_mpp)
        if diff < min_diff:
            min_diff = diff
            closest_level = level

    return closest_level
