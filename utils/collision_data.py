from utils.memory import read_u32, read_vector_3d

DEFAULT_KCL_PATH = 'desert_course/src/course_collision.kcl'

header_offset = 0x3c

def read_collision_data(kcl_path: str = DEFAULT_KCL_PATH):
    with open(kcl_path, 'rb') as f:
        content = f.read()
        pos_offset = read_u32(content, 0x00)
        section_pos_1 = read_u32(content, 0x04)
        section_pos_2 = read_u32(content, 0x08)
        section_pos_3 = read_u32(content, 0x0C)
        pos_end = min(section_pos_1, section_pos_2, section_pos_3)
        pos_data = content[pos_offset:pos_end]
        
        points = []
        for i in range(0, len(pos_data), 0x0C):
            x, y, z = read_vector_3d(pos_data, i)
            points.append([x, y, z])
            
        return points
    
if __name__ == "__main__":
    points = read_collision_data()
    print(points)