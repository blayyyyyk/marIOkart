# Training a model to play Mariokart DS

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
Course data is located as an `.nkm` file within the MKDS ROM. In order the retreive this file, we need to unpack the ROM file. I used [kiwi.ds](https://projectpokemon.org/home/files/file/2073-nds-editor-kiwids/) to retreive the course files. The course files are compressed as `.carc`. To uncompress it, I used [NArchive](https://github.com/nickworonekin/narchive/tree/master/src/Narchive) to extract the course files. I'm mainly focused on reading the checkpoint data for a course, so I ignore the rest of the files except the `.nkm` files.

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

