def filter_cond(line_dict):
    """Filter function
    Takes a dict with field names as argument
    Returns True if conditions are satisfied
    """
    try:
        val = int(line_dict["if1"])
        cond_match = (val > 20 and val < 40)
        return cond_match
    except:
        return False
