from utils.memory import read_vector_2d, read_fx32, read_u16, read_u32

DEFAULT_NKM_PATH = 'desert_course/src/course_map.nkm'

header_offset = 0x4C
    
def get_section_header(data, addr=0x00):
    title = data[addr:addr+0x04]
    title = title.decode('ascii')
    num_entries = read_u32(data, addr+0x04)
    return title, num_entries, data[addr+0x08:]

def read_checkpoint_data(nkm_path: str = DEFAULT_NKM_PATH):
    with open(nkm_path, 'rb') as file:
        content = file.read()
        checkpoint_offset = read_u32(content, 0x2C) + header_offset
        checkpoint_end = read_u32(content, 0x30) + header_offset
        checkpoint_data = content[checkpoint_offset:checkpoint_end]
        _, num_entries, checkpoint_data = get_section_header(checkpoint_data)
        
        entries = []
        for i in range(num_entries):
            start = i*0x24
            end = start+0x24
            entry_data = checkpoint_data[start:end]
            entry_dict = {
                "p0": read_vector_2d(entry_data, 0x00),
                "p1": read_vector_2d(entry_data, 0x08),
                "distance": read_fx32(entry_data, 0x18),
                "id": read_u16(entry_data, 0x20),
            }
            entries.append(entry_dict)
            
        return entries
        

if __name__ == "__main__":
    entries = read_checkpoint_data()
    print(entries)