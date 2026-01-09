nm -gU private/desmume_installs/libdesmume.dylib

0000000006437dc8 S _FeedbackON
0000000003120e98 S _GBACartridge_RomPath
0000000003120eb0 S _GBACartridge_SRAMPath
0000000000fbdcc0 S _GPU
000000000311bad0 S _InputDisplayString
00000000031208d8 S _LagFrameFlag
0000000000fc1200 S _MMU                      <-
00000000031164a0 S _MMU_new
0000000003116b00 S _MMU_timing
00000000031208e4 S _MicSampleSelection
00000000003b8fc0 S _NDS_ARM7
00000000003b9150 S _NDS_ARM9
... symbols omitted ...
00000000000068e0 T _desmume_memory_get_next_instruction
0000000000003e88 T _desmume_memory_read_byte
00000000000040e0 T _desmume_memory_read_byte_signed
0000000000004918 T _desmume_memory_read_long <-
0000000000004d4c T _desmume_memory_read_long_signed
00000000000064ac T _desmume_memory_read_register
0000000000004330 T _desmume_memory_read_short
0000000000004628 T _desmume_memory_read_short_signed