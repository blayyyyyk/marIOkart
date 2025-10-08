# Training a model to play Mariokart DS

##### Figure 1: Demo of game overlay rendering in realtime using Pycairo the Python GTK library. NOTE: AI demo coming soon!
![overlay_demo_clip_1.gif](https://github.com/gg-blake/marIOkart/blob/main/media/overlay_demo_clip_1.gif?raw=True)

```
pip install -r requirements.txt
```

# Overview
The project falls down into three main parts
1) Reverse engineering MKDS Rom for car data (Done)
2) Making a visualization for debugging and benchmarking (Done)
3) Building and training the model (In Progess)

# Reverse Engineering Mariokart DS
The reverse engineering process consisted mainly of using [dynamic code analysis](https://en.wikipedia.org/wiki/Dynamic_program_analysis) to extract import values from kart's game state. Values consisted of but were not limited to:
- Car's position
- Car's orientation
- Car's acceleration/Speed
- Car's collected ability
- Enemy positions

## Interpreting WRAM
I used an emulation tool called [Desmume](https://desmume.org/) to run the MKDS rom on my PC. Desmume comes with built in features for watching the memory which I utilized heavily. Additionally, my ML interface utilized [py-desmume](https://py-desmume.readthedocs.io/en/latest/), a python interface for using desmume's frontend. For US versions of MKDS, the pointer to the kart's race data is located at `0x0217ACF8` in

The DS uses a dual CPU system with ARM9 and ARM7 CPUs. Although memory is divided between the two (i.e. they share the same address space), the data that I care about is stored exclusively on the ARM9 CPU's memory. In Mariokart DS, the game state is stored on the Main RAM of the ARM9 CPU from `0x02000000` to `0x023FFFFF`, in [little endian](https://en.wikipedia.org/wiki/Endianness) format. 

### Kart Data
The pointer to the kart's game data (position, orientation, powerup, etc.) is located at `0x0217ACF8`, stored as an unsigned 32-bit integer. Starting at the kart's game data in memory, kart values of interest live at the following offsets:
- position vector (`0x80`) (3x `fx32`)
- movement direction vector (`0x68`) (3x `fx32`)
Many game values including vectors are stored as fixed point data types according to the [NitroSDK/NitroMath spec](https://twlsdk.randommeaninglesscharacters.com/docs/nitro/NitroSDK/fx/list_fx.html). Unlike floating point values that have an exponent component to the bit sequence, Mario Kart DS's fixed point datatypes, specifically, `fx32` have a dedicated 1-bit sign, 19-bit integer, and 12-bit fraction.

### Camera Data
The pointer to the car camera's data is located at `0x0217AA4C`, stored as an unsigned 32-bit integer. Starting at the car camera's data in memory, camera values of interest live at the following offsets:
- position vector (`0x24`) (3x `fx32`)
- elevation (`0x178`) (1x `fx32`)
- target position vector (`0x18`) (3x `fx32`)
- field of view (`0x60`) (1x `u16`)
- aspect ratio (`0x6C`) (1x `fx32`)
This is all the camera data that we need to reconstruct the camera's [perspective projection and model view matrices](https://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#the-model-view-and-projection-matrices). This will come in handy when I discuss the visualization overlay later on.

## Interpreting Course Data

For course information, we have two main files of interest per course: KCL and NKM files. I used [kiwi.ds](https://projectpokemon.org/home/files/file/2073-nds-editor-kiwids/) to retreive the course files. The course files are compressed as `.carc`. To uncompress it, I used [NArchive](https://github.com/nickworonekin/narchive/tree/master/src/Narchive) to extract the course files.

### KCL
The course's KCL file (ending in `.kcl`) contains information about the course's collision data. This will help us compute the information we need to pass to our model for obstacle avoidance.

The file contains a header section specifying section offsets as well four data sections: the vertex positions, the normal vectors of the triangles that make up the course, the triangular prism data, and the spatial index. The triangular prism entries each contain an index to the position section, three indices to the normal vector section for the edge normal vectors, a fourth normal index for the triangle's face normal, prism , and a triangle attribute data. The triangle's other two vertices can be reconstructed with the following Pytorch code:
```python
def compute_triangles(self):
    # Indexed Vectors
    height = self.prisms.height
    vertex_0 = self.positions[self.prisms.pos_i]
    face_norm = self.normals[self.prisms.fnrm_i]
    edge_norm_0 = self.normals[self.prisms.enrm1_i]
    edge_norm_1 = self.normals[self.prisms.enrm2_i]
    edge_norm_2 = self.normals[self.prisms.enrm3_i]

    cross_a = edge_norm_0.cross(face_norm, dim=-1)
    cross_b = edge_norm_1.cross(face_norm, dim=-1)

    vertex_1 = (
        vertex_0
        + cross_b * (height / torch.linalg.vecdot(edge_norm_2, cross_b))[:, None]
    )
    vertex_2 = (
        vertex_0
        + cross_a * (height / torch.linalg.vecdot(edge_norm_2, cross_a))[:, None]
    )

    out = torch.stack([vertex_0, vertex_1, vertex_2], dim=1)

    return out
```

The prism block data section is a spatial octree data structure which enables fast collision detection, which in most cases is the most computationally expensive task in game development. I reimplemented the search algorithm for searching for a prism containing an arbitrary point below.

```python
def search_block(self, point: tuple[float, float, float] | list[float, float, float]):
    """
    Return the offset of the leaf node containing a queried point
    """
    block_start = self.block_data_offset
    block_data = self.data[block_start:]
    
    px, py, pz = point
    minx, miny, minz = self.area_min_pos

    x = int(px - minx)
    if (x & self.area_x_width_mask) != 0:
        return None

    y = int(py - miny)
    if (y & self.area_y_width_mask) != 0:
        return None

    z = int(pz - minz)
    if (z & self.area_z_width_mask) != 0:
        return None

    # initialize root
    shift = self.block_width_shift
    cur_block_offset = 0  # root at start of block_data

    index = 4 * (((z >> shift) << self.area_xy_blocks_shift)
        | ((y >> shift) << self.area_x_blocks_shift)
                | (x >> shift))

    while True:
        offset = read_u32(block_data, cur_block_offset + index)

        if (offset & 0x80000000) != 0:
            # negative flag = leaf node
            break

        shift -= 1
        cur_block_offset += offset

        # initialize next index
        index = 4 * (((z >> shift) & 1) << 2
                    | ((y >> shift) & 1) << 1
                    | ((x >> shift) & 1))

    # leaf = return pointer into block_data (as slice)
    leaf_offset = cur_block_offset + (offset & 0x7FFFFFFF)
    return leaf_offset
```

This search algorithm gives an offset into the section of prism group containing the point. The offset beginning and terminating with `0` specifies an array of indices into the triangular prisms section. Combining both the search and triangular prism reconstruction, we can finding the group of prisms containing a given point.

```python
def search_triangles(
    self,
    point: tuple[float, float, float] | torch.Tensor,
    filter_attribute_id: int | None = None,
):
    assert self.triangles is not None
    if not isinstance(point, tuple):
        p = tuple(point.tolist())
        leaf_offset = self.search_block(p)
    else:
        leaf_offset = self.search_block(point)

    if leaf_offset is None:
        return None

    tri_indices: list[int] = []
    chunk_size = 0x02
    start = self.block_data_offset + leaf_offset + chunk_size
    for data_offset in range(start, len(self.data), chunk_size):
        idx = read_u16(self.data, data_offset) - 1
        if idx == -1:
            break

        tri_indices.append(idx)

    if len(tri_indices) == 0:
        return None

    return tri_indices
```

Prism attribute data provides a plethora of prism information such as the prisms collision type. Now we can filter for prisms that are only floor prisms, wall prisms, offroad prisms and a lot more I haven't looked into. This is important for passing in distance information into our model since I'm assuming that floors/roads will never be an obstacle to avoid.

### NKM
The course's NKM file (ending in `.nkm`) contains information about  
Course data is located as an `.nkm` file within the MKDS ROM. In order the retreive this file, we need to unpack the ROM file. I'm mainly focused on reading the checkpoint data for a course, so I ignore the rest of the files except the `.nkm` files.

NKM files are essentially specialized bin files. The spec for this file can be found [here](https://wiki.tockdom.com/wiki/NKM_(File_Format)#cite_note-MoreCPOIInfo-4). The NKM file has a header that specifies the byte offset of each data section in the file. The `CPOI` section contains all the entries for checkpoints on a map. It's section offset is found at `0x2C`. Each data section specifies it name and the number of entries. `CPOI` entries are 36 bytes in size and contain:
1) left position vector (`0x00`) (2x `fx32`)
2) right position vector (`0x08`) (2x `fx32`)
...
5) distance (`0x18`) (1x `fx32`)
...
8) key id (`0x20`) (1x `u16`)
9) respawn id (`0x22`) (1x `u8`)
...
Having this data is useful since I can use this to calculate the player's forward facing distance to wall, assuming the checkpoints are positioned within the bounds of the map.

## Building the overlay

Using the camera data that I collected from the desmume memory buffer, I constructed a OpenGL-like projection matrix. Both Nitro SDK and OpenGL use right-handed coordinate systems.

*Projection Matrix*
$$
\mathbf{P} =
\begin{bmatrix}
\frac{1}{\tan(fov) \, \text{aspect}} & 0 & 0 & 0 \\
0 & \frac{1}{\tan(fov)} & 0 & 0 \\
0 & 0 & \frac{far + near}{near - far} & -\frac{2 \, far \, near}{near - far} \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

The fov collected from memory is already the actual fov divided by 2, so its presence in the tan functions is omitted.

I also read constructed my model view matrix by assuming an arbitrary up vector and a right vector that is perpendicular. The **model–view matrix** transforms world-space coordinates into camera-space coordinates.

$$
\begin{align}
\mathbf{f} &= \frac{\mathbf{c_t} - \mathbf{c_p}}{ \| \mathbf{c_t} - \mathbf{c_p} \| } \\
\mathbf{R} &= \text{compute\_orthonormal\_basis}(\mathbf{f}) \\
\mathbf{p}_{proj} &= \mathbf{R} \, \mathbf{c_p} \\
\mathbf{M}_{view} &=
\begin{bmatrix}
\mathbf{R} & -\mathbf{R} \, \mathbf{c_p} \\
\mathbf{0}^T & 1
\end{bmatrix}
\end{align}
$$

I with my model view and projection matrices, I can now project a 3d-point in game to a 2d point on the screen. I can also compute a depth dimension for making far points small and close points large.

$$
\mathbf{p}_{clip} = \mathbf{P} \, \mathbf{M}_{view} \, \mathbf{p}_{world}
$$

$$
\mathbf{p}_{ndc} = \frac{\mathbf{p}_{clip, xyz}}{\mathbf{p}_{clip, w}}
$$

$$
\begin{align}
x_{screen} &= \frac{(x_{ndc} + 1)}{2} \, W_{screen} \\
y_{screen} &= \frac{(1 - y_{ndc})}{2} \, H_{screen}
\end{align}
$$

Taking all the spatial information I've collected from the KCL and NKM files, and the game memory buffer, I can now make a fully working game overlay. I used Pycairo to render simple shapes to a an existing GTK window containing the emulator game.

##### Figure 1: Demo of game overlay rendering in realtime using Pycairo the Python GTK library. NOTE: AI demo coming soon!
![overlay_demo_clip_1.gif](https://github.com/gg-blake/marIOkart/blob/main/media/overlay_demo_clip_1.gif?raw=True)

# Building the game model

This task is ongoing :)
