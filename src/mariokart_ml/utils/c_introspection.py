import ctypes

from desmume.mkds.mkds import POINTER_T, struct_driver_t


def get_field_at_offset(struct_class, target_offset):
    """
    Returns the name of the field that occupies the given byte offset.
    """
    # Iterate through the defined fields
    for field_name, field_type in struct_class._fields_:
        # Access the field object from the class to get its offset
        field_obj = getattr(struct_class, field_name)
        start = field_obj.offset
        size = ctypes.sizeof(field_type)
        end = start + size

        # Check if the target offset falls within this field's range
        if start <= target_offset < end:
            return field_name

    return None

def main():
    import sys

    offset = int(sys.argv[1], 16)
    struct_class = struct_driver_t

    field_name = get_field_at_offset(struct_class, offset)
    print(f"Offset {hex(offset)} is in field: {field_name}")


if __name__ == "__main__":
    main()
