def map_field_str2int(map_field_str):
    usual_fields = "IQU"
    if map_field_str not in usual_fields:
        raise ValueError(f"map_field_str must be one of {usual_fields}")
    return usual_fields.index(map_field_str)
