from .utils import (
    read_u8,
    read_u16,
    read_u32,
    read_s16,
    read_s32,
    read_fx16,
    read_fx32,
    read_vector_2d_fx32,
    read_vector_3d_fx32
)


class Section:
    """
    Base class for sections in an NKM file.

    Overview
    --------
    Most NKM sections (all except STAG) begin with an 8-byte section header:
      - 0x00 (4 bytes): Section magic (ASCII, e.g., "OBJI").
      - 0x04 (4 bytes): Number of entries in the section (UInt32).
    Immediately after the header the section contains `entry_count` entries,
    each with a fixed stride (size) which is specified by the concrete Section
    subclass.

    This base class encapsulates:
      - raw `data` for the whole section (header + entries)
      - `stride` (size in bytes of a single entry)
      - `entry_count` parsed from the header
      - an iterator over each fixed-size entry slice

    Notes / Caveats
    ----------------
    * This class assumes the `data` passed includes the section header.
    * If a section contains variable-length entries (rare in NKM), a custom
      parser should be used instead of relying on a fixed stride.
    * Many sections have fields where 0xFFFF or 0xFF indicates "unused" or
      "none" — callers should treat those sentinel values accordingly.
    """
    def __init__(self, data, size):
        self.data = data
        self.iter = range(0x08, len(data), size)
        self.stride = size
        # Number of entries (UInt32 at offset 0x04 of the section header).
        # If `data` is not the whole section starting with the header, this
        # will be incorrect — ensure `data` includes the 8-byte section header.
        self.entry_count = read_u32(data, 0x04)

    def __len__(self):
        return self.entry_count

    def __iter__(self):
        for i in self.iter:
            yield self.data[i:i+self.stride]


class OBJI(Section):
    """
    OBJI — Object Instances section.

    Stride: 0x3C bytes per entry.

    Purpose
    -------
    Describes every object placed in the track: visual decorations, interactive
    objects, obstacle instances, item boxes, etc. These objects are instantiated
    by the game engine and can be linked to PATH routes.

    Offsets (per-entry)
    -------------------
    0x00: Vector (VecFx32) — 3D position (X, Y, Z). Units are FX32 format.
    0x0C: Vector — 3D rotation vector (often treated as Euler-like or direction).
    0x18: Vector — 3D scale vector.
    0x24: UInt16 — Object ID (index into object database). Value selects the
                     model/behavior (see community object lists).
    0x26: UInt16 — Route ID (0xFFFF indicates "no route").
    0x28..0x37: UInt32[4] — Object-specific settings (four 32-bit words). Their
                          meaning depends entirely on the object ID.
    0x38: UInt32 — Show in Time Trials flag (1 = visible, 0 = hidden).

    Gameplay Context
    ----------------
    - The object ID decides both model and in some cases runtime logic (collision,
      activatable behaviors).
    - The route_id links the object to a PATH. If present, the object will be
      moved/animated by the PATH's POIT points at runtime (e.g., moving platforms,
      cameras).
    - The object settings are used by many object types to control behavior:
      rotation speed, spawn flags, timers, random seeds, or other per-object
      parameters. Different object IDs require different decoding logic.

    Reverse-Engineering Notes / Tips
    --------------------------------
    - The four 32-bit settings differ radically between object types — community
      wikis often contain per-object decode rules (use object ID to branch).
    - The presence of a non-0 or non-0xFFFFFFFF route_id often means the object
      will be animated along that route; route indices are bytes in PATH entries
      but stored here as a 16-bit value (keep an eye on sign/width).
    - Show-in-time-trials: historically some editors used 0/1 inverse; verify
      against known tracks when in doubt.

    Parsing Caveat
    --------------
    We read `object_id` as `read_u16` and `route_id` as `read_u16`. Consumers of
    this class should treat 0xFFFF in `route_id` as "no route".
    """
    def __init__(self, data):
        super().__init__(data, 0x3C)
        self.rot_vec1 = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.rot_vec2 = [read_vector_3d_fx32(d, 0x0C) for d in self]
        self.scale_vec = [read_vector_3d_fx32(d, 0x18) for d in self]
        self.object_id = [read_u16(d, 0x24) for d in self]
        self.route_id = [read_u16(d, 0x26) for d in self]
        self.object_settings = [
            [read_u32(d, j) for j in range(0x28, 0x38, 0x04)] for d in self
        ]
        self.show_in_time_trials = [read_u32(d, 0x38) for d in self]


class PATH(Section):
    """
    PATH — Path metadata.

    Stride: 0x04 bytes per entry.

    Purpose
    -------
    Describes metadata for routes used by objects and cameras. Each PATH entry
    points to a sequence of POIT entries (control points) that define the route.

    Offsets (per-entry)
    -------------------
    0x00: Byte — Route ID.
    0x01: Byte — Loop flag (1 if the route loops, 0 otherwise).
    0x02: UInt16 — Number of points for the route (number of POIT entries).

    Gameplay Context
    ----------------
    - Objects or cameras that reference a route will move along the POIT points
      in order; if the loop flag is set, the route repeats.
    - Route ID is commonly a small integer; the PATH list enumerates all routes
      in use on the track.

    Reverse-Engineering Notes / Caveats
    -----------------------------------
    * The canonical spec states: 0x01 == 1 if the route loops, 0 otherwise.
      In the code you originally used `read_u8(d, 0x01) != 1` (which inverts
      the meaning). I have NOT altered your logic — but be aware of the
      discrepancy: callers should expect `True` when the route loops.
      Consider changing to `read_u8(... ) == 1` for clarity.
    * `point_count` enumerates POIT entries but the mapping from global POIT
      index to route is (index offset + length) — consumers should reconstruct
      the actual POIT index ranges using CPAT/EPAT/IPAT/MEPA grouping sections
      when applicable (these groupings partition points).
    """
    def __init__(self, data):
        super().__init__(data, 0x04)
        self.route_id = [read_u8(d, 0x00) for d in self]
        # Warning: original code used inverted logic. The spec: 1 means loop.
        # We preserve original behavior here; consider flipping if you want literal.
        self.has_loop = [read_u8(d, 0x01) != 1 for d in self]
        self.point_count = [read_u16(d, 0x02) for d in self]


class POIT(Section):
    """
    POIT — Path points (control points).

    Stride: 0x14 bytes per entry.

    Purpose
    -------
    Stores the actual 3D points used by PATH routes. Points are grouped by route
    using the counts stored in PATH plus the various *PAT grouping sections.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position (VecFx32).
    0x0C: Byte — Point index in the route (order index).
    0x0D: Byte — Unknown / padding.
    0x0E: Int16 — Point duration (signed). Not always used; may control timing
                   between points (for camera/object interpolation).
    0x10: UInt32 — Unknown (possibly reserved or flags).

    Gameplay Context
    ----------------
    - Moving objects and cameras read these points sequentially to interpolate
      positions. The "point_index" normally indicates position in the route's
      ordering (0,1,2,...).
    - Some cameras or scripted objects use `point_duration` to wait between
      points, enabling non-linear motion.
    - POIT order in the file is important: group membership is often determined
      by successive ranges; use PAT sections to map ranges to specific routes.

    Reverse-Engineering Notes
    -------------------------
    - The unknown fields often show consistent patterns per track editor —
      check community resources to decode them for special behaviors.
    - Some older track versions or beta files encode rotation differently; when
      in doubt, cross-check with KTPJ notes for version-dependent behavior.
    """
    def __init__(self, data):
        super().__init__(data, 0x14)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.point_index = [read_u8(d, 0x0C) for d in self]
        self.unknown1 = [read_u8(d, 0x0D) for d in self]
        self.point_duration = [read_s16(d, 0x0E) for d in self]
        self.unknown2 = [read_u32(d, 0x10) for d in self]


class STAG:
    """
    STAG — Stage (track) information.

    Fixed-size: The STAG section is unique in NKM: it does NOT have a section
    header and is a single 0x2C-byte structure placed directly in the file
    (after POIT in the canonical header ordering).

    Purpose
    -------
    Contains global track settings: track ID, default lap count, fog settings,
    colors used by KCL (collision visual palettes), and other miscellaneous bytes.

    Offsets / Fields (STAG 0x00..0x2B)
    ---------------------------------
    0x00: String (4 bytes) — Section magic "STAG" (present in file).
    0x04: UInt16 — Track ID.
    0x06: UInt16 — Amount of laps.
    0x08: Byte — Unknown (seen set to small integers).
    0x09: Byte — Fog enabled (1 = enabled, 0 = disabled).
    0x0A: Byte — Fog table generation mode.
    0x0B: Byte — Fog slope.
    0x0C..0x13: Byte[8] — Unknown region (padding or extra flags).
    0x14: Fx32 — Fog distance (fixed-point).
    0x18: GXRgb — Fog color (not yet implemented).
    0x1A: UInt16 — Fog alpha (0..15 typical range).
    0x1C: GXRgb — KCL color 1.
    0x1E: GXRgb — KCL color 2.
    0x20: GXRgb — KCL color 3.
    0x22: GXRgb — KCL color 4.
    0x24..0x2B: Byte[8] — Another unknown block.

    Gameplay Context
    ----------------
    - Amount of laps controls how many laps the race uses by default for the
      stage; some tracks use lap_count = 0 for special cases (verify per track).
    - Fog parameters influence rendering: enabling fog can hide distant objects
      and alter perceived depth; fog color & distance control atmosphere.
    - KCL colors are the default palette for collision visualization (useful
      for editors and collision debugging).

    Implementation Notes
    --------------------
    * This class sets placeholders for color fields (GXRgb) — you can implement
      `GXRgb` decoding (usually 2 bytes per color or platform-specific) and
      populate these fields for richer output.
    * The unknown arrays are repeated in code (unknown2 and unknown3) — that
      mirrors the spec layout but may be redundant; keep one copy or rename for
      clarity if desired.
    """
    def __init__(self, data):
        self.track_id = read_u16(data, 0x04)
        self.amt_of_laps = read_u16(data, 0x06)
        self.unknown1 = read_u8(data, 0x08)
        self.fog_enabled = read_u8(data, 0x09) != 0
        self.fog_table_generation_mode = read_u8(data, 0x0A)
        self.fog_slope = read_u8(data, 0x0B)
        self.unknown2 = [read_u8(data, 0x0C + i) for i in range(0, 0x14, 0x01)]
        self.fog_distance = read_fx32(data, 0x14)
        self.fog_color = None  # TODO: Implement GXRgb parsing & set this.
        self.fog_alpha = read_u16(data, 0x1A)
        self.kcl_color1 = None  # TODO: Implement GXRgb
        self.kcl_color2 = None
        self.kcl_color3 = None
        self.kcl_color4 = None
        self.unknown3 = [read_u8(data, 0x0C + i) for i in range(0, 0x14, 0x01)]


class KTPS(Section):
    """
    KTPS — Kart/Start Positions (Start points for racers).

    Stride: 0x1C bytes per entry.

    Purpose
    -------
    Defines start positions (spawn/starting grid) for players/racers. Typically
    used for the main race starts; can also be used in battle or mission modes.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position (VecFx32) for the starting spawn.
    0x0C: Vector — 3D rotation vector (direction the racer initially faces).
    0x18: UInt16 — Padding (usually 0xFFFF).
    0x1A: UInt16 — Start position index (used in battle/mission mode; 0xFFFF
                       for normal race courses).

    Gameplay Context
    ----------------
    - On race start, the game picks starting positions from this section.
    - `start_position_index` is relevant for battle stages or mission mode where
      start ordering differs from main racing.
    - The rotation vector might be stored differently in beta versions — the
      canonical community notes indicate the Y-rotation sometimes needs to be
      computed via Atan2 on rotation vector components for older versions.

    Notes / Community Tips
    ----------------------
    - Typically the number of KTPS entries equals the number of players or more
      (some tracks list more possible starts than vehicles).
    - If you want to reposition start locations, modify positions and write the
      file back in FX32 format.
    """
    def __init__(self, data):
        super().__init__(data, 0x1C)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.rot_vec = [read_vector_3d_fx32(d, 0x0C) for d in self]
        self.padding = [read_u16(d, 0x18) for d in self]
        self.start_position_index = [read_u16(d, 0x1A) for d in self]


class KTPJ(Section):
    """
    KTPJ — Respawn positions (kart respawn).

    Stride: 0x20 bytes per entry.

    Purpose
    -------
    Positions to which a kart (player) can respawn after falling off or during
    certain scripted events. Contains references to enemy/item points (EPOI/IPOI)
    to determine nearby behavior or AI context.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position vector.
    0x0C: Vector — 3D rotation vector.
    0x18: UInt16 — Enemy position ID (EPOI index).
    0x1A: UInt16 — Item position ID (IPOI index).
    0x1C: UInt32 — Respawn ID (not present in the oldest beta versions).

    Gameplay Context
    ----------------
    - When a kart falls off the track or hits a severe collision, the engine
      picks a KTPJ respawn that matches the current lap and nearby conditions.
    - The enemy/item IDs let respawn logic pick nearby AI/item spawn points for
      smoother reintroduction.

    Version Notes
    -------------
    - For version 0x1E (older beta), the final Respawn ID (0x1C) may not exist.
    - The rotation encoding changed for early beta versions; if reading older
      tracks, compute Y-rotation via atan2(Rx, Rz) to convert to degrees.

    Remarks for Parser
    ------------------
    * We read `respawn_id` as a 32-bit value; if parsing older tracks that omit
      it, ensure you guard read beyond section length.
    """
    def __init__(self, data):
        super().__init__(data, 0x20)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.rot_vec = [read_vector_3d_fx32(d, 0x0C) for d in self]
        self.enemy_position_id = [read_u16(d, 0x18) for d in self]  # EPOI reference
        self.item_position_id = [read_u16(d, 0x1A) for d in self]   # IPOI reference
        # Respawn ID may not exist in very old versions; this read assumes it does.
        self.respawn_id = [read_u32(d, 0x1C) for d in self]


class KTP2(Section):
    """
    KTP2 — Lap checkpoints (points to pass to count lap progress).

    Stride: 0x1C bytes per entry.

    Purpose
    -------
    Defines the "lap gate" points that the engine checks to determine whether a
    player completed a lap. Usually combined with timing/ordering checks.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position vector.
    0x0C: Vector — 3D rotation vector.
    0x18: UInt16 — Padding (0xFFFF).
    0x1A: UInt16 — Index (commonly 0xFFFF).

    Gameplay Context
    ----------------
    - The race logic queries these points as canonical lap markers.
    - Often unused fields are set to 0xFFFF; do not treat them as valid indices.

    Notes
    -----
    - The "Index" is usually unused (set to 0xFFFF); if present, it may
      participate in specialized lap logic or developer tools.
    """
    def __init__(self, data):
        super().__init__(data, 0x1C)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.rot_vec = [read_vector_3d_fx32(d, 0x0C) for d in self]
        self.padding = [read_u16(d, 0x18) for d in self]
        self.index = [read_u16(d, 0x1A) for d in self]


class KTPC(Section):
    """
    KTPC — Cannon / Pipe destination points.

    Stride: 0x1C bytes per entry.

    Purpose
    -------
    Describes destinations for cannons/pipes used in certain stages (mostly
    in battle stages). Cannon indices are used by specialized collision types.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position vector (destination).
    0x0C: Vector — 3D rotation vector.
    0x18: UInt16 — Unknown field (investigate for specific behaviors).
    0x1A: UInt16 — Cannon index (used by 'Cannon Activator' collision types).

    Gameplay Context
    ----------------
    - Cannon/pipe logic teleports an object/player to the KTPC destination.
    - The cannon_index can be used as a link between activator and destination.

    Notes
    -----
    - In many tracks this section is small or absent; treat missing sections
      gracefully when writing tools that modify KTPC entries.
    """
    def __init__(self, data):
        super().__init__(data, 0x1C)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.rot_vec = [read_vector_3d_fx32(d, 0x0C) for d in self]
        self.unknown = [read_u16(d, 0x18) for d in self]
        self.cannon_index = [read_u16(d, 0x1A) for d in self]


class KTPM(Section):
    """
    KTPM — Mission points.

    Stride: 0x1C bytes per entry.

    Purpose
    -------
    Points used by mission objectives (mission mode). They often resemble
    KTPS/KTP2 entries but have mission-specific indexing.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position vector.
    0x0C: Vector — 3D rotation vector.
    0x18: UInt16 — Padding (0xFFFF).
    0x1A: UInt16 — Index (used to map to mission logic).

    Gameplay Context
    ----------------
    - Missions may use these points to define target positions or spawns.
    - `index` can be used in mission scripts to select a specific point
      out of the KTPM array.

    Notes
    -----
    - If the track is not a mission type, these often default to 0xFFFF.
    """
    def __init__(self, data):
        super().__init__(data, 0x1C)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.rot_vec = [read_vector_3d_fx32(d, 0x0C) for d in self]
        self.padding = [read_u16(d, 0x18) for d in self]
        self.index = [read_u16(d, 0x1A) for d in self]


class CPOI(Section):
    """
    CPOI — Checkpoints (2D oriented).

    Stride: 0x24 bytes per entry.

    Purpose
    -------
    Defines checkpoint segments used for lap counting, key handling, and
    respawn logic. CPOI includes two 2D positions and precomputed trig/distance
    values used by the engine to determine crossing and checkpoint ordering.

    Offsets (per-entry)
    -------------------
    0x00: Vector2D — Position 1 (VecFx32 2D).
    0x08: Vector2D — Position 2 (VecFx32 2D) — typically the other edge of checkpoint.
    0x10: Fx32 — Precomputed sinus (used for orientation math).
    0x14: Fx32 — Precomputed cosinus.
    0x18: Fx32 — Distance between position1 and position2 (checkpoint length).
    0x1C: Int16 — Section data 1 (unknown; may encode connectivity).
    0x1E: Int16 — Section data 2 (unknown).
    0x20: UInt16 — Key ID:
             0x0000 = lap counter,
             0x0001..0xFFFE = keyed checkpoint,
             0xFFFF = no key.
    0x22: Byte — Respawn ID.
    0x23: Byte — Unknown/padding.

    Gameplay Context
    ----------------
    - CPOIs are the canonical gate the engine checks for lap progress.
    - The Key ID allows some checkpoints to behave as keys (for gates, or race
      logic). If Key ID == 0x0000 the point counts as the lap marker.
    - The precomputed sinus/cosinus/distance allow the engine to quickly test
      whether a player crossed the checkpoint along the correct orientation.

    Reverse-Engineering Notes
    -------------------------
    - Community docs note the `section_data` fields are partially decoded for
      special track logic. If you need precise behavior, compare original
      tracks and observe in-engine behavior.
    """
    def __init__(self, data):
        super().__init__(data, 0x24)
        self.position1 = [read_vector_2d_fx32(d, 0x00) for d in self]
        self.position2 = [read_vector_2d_fx32(d, 0x08) for d in self]
        self.sinus = [read_fx32(d, 0x10) for d in self]
        self.cosinus = [read_fx32(d, 0x14) for d in self]
        self.distance = [read_fx32(d, 0x18) for d in self]
        self.section_data1 = [read_u16(d, 0x1C) for d in self]
        self.section_data2 = [read_u16(d, 0x1E) for d in self]
        self.key_id = [read_u16(d, 0x20) for d in self]
        self.respawn_id = [read_u8(d, 0x22) for d in self]
        self.unknown = [read_u8(d, 0x23) for d in self]


class CPAT(Section):
    """
    CPAT — CPOI grouping (checkpoint groups).

    Stride: 0x0C bytes per entry.

    Purpose
    -------
    Groups CPOI entries into logical sequences (routes/sections). Each CPAT
    entry points to a contiguous block of CPOI points and describes adjacency
    (previous/next groups) and section order.

    Offsets (per-entry)
    -------------------
    0x00: UInt16 — Point start index into global CPOI array.
    0x02: UInt16 — Point length (number of CPOI points).
    0x04..0x06: Byte[3] — Next group indices (up to 3). 0xFF = unused.
    0x07..0x09: Byte[3] — Previous group indices (up to 3). 0xFF = unused.
    0x0A: Int16 — Section order (signed; determines ordering among groups).

    Gameplay Context
    ----------------
    - CPAT allows complex checkpoint graphs (non-linear courses) by connecting
      multiple CPOI groups.
    - Useful when a single lap uses multiple discontiguous checkpoint regions.

    Implementation Notes
    --------------------
    - The `next_group` and `prev_group` arrays use 0xFF as "unused"; treat
      sentinel values accordingly.
    """
    def __init__(self, data):
        super().__init__(data, 0x0C)
        self.point_start = [read_u16(d, 0x00) for d in self]
        self.point_length = [read_u16(d, 0x02) for d in self]
        self.next_group = [
            [read_u8(d, i) for i in range(0x04, 0x07, 0x01)]
            for d in self
        ]
        self.prev_group = [
            [read_u8(d, i) for i in range(0x07, 0x0A, 0x01)]
            for d in self
        ]
        self.section_order = [read_s16(d, 0x0A) for d in self]


class IPOI(Section):
    """
    IPOI — Item spawn points.

    Stride: 0x14 bytes per entry.

    Purpose
    -------
    Describes where items (like red shells, bananas) may spawn or where items
    follow a path along a route. IPOI entries are often referenced by KTPJ
    (respawn) or object logic.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position vector.
    0x0C: Fx32 — Point scale (fixed point value, used for size or weighting).
    0x10: UInt32 — Unknown; reserved or object-specific.

    Gameplay Context
    ----------------
    - Items may spawn at IPOI locations or be used for scripted item routes.
    - `point_scale` may modify spawn probability or area radius.

    Notes
    -----
    - The unknown 32-bit value is often zero; it may contain bitflags in
      certain custom tracks.
    """
    def __init__(self, data):
        super().__init__(data, 0x14)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.point_scale = [read_fx32(d, 0x0C) for d in self]
        self.unknown = [read_u32(d, 0x10) for d in self]


class IPAT(Section):
    """
    IPAT — IPOI grouping.

    Stride: 0x0C bytes per entry.

    Purpose
    -------
    Group IPOI entries into contiguous point ranges and define adjacency, much
    like CPAT but for item points.

    Offsets (per-entry)
    -------------------
    0x00: UInt16 — Point start index into IPOI array.
    0x02: UInt16 — Point length.
    0x04..0x06: Byte[3] — Next group indices (0xFF unused).
    0x07..0x09: Byte[3] — Previous group indices (0xFF unused).
    0x0A: Int16 — Section order.

    Gameplay Context
    ----------------
    - Used by item routing and by respawn selection to find nearby item points.
    """
    def __init__(self, data):
        super().__init__(data, 0x0C)
        self.point_start = [read_u16(d, 0x00) for d in self]
        self.point_length = [read_u16(d, 0x02) for d in self]
        self.next_group = [
            [read_u8(d, i) for i in range(0x04, 0x07, 0x01)]
            for d in self
        ]
        self.prev_group = [
            [read_u8(d, i) for i in range(0x07, 0x0A, 0x01)]
            for d in self
        ]
        self.section_order = [read_s16(d, 0x0A) for d in self]


class EPOI(Section):
    """
    EPOI — Enemy/CPU path points.

    Stride: 0x18 bytes per entry.

    Purpose
    -------
    Defines points that the CPU opponents use for their routing (AI paths).
    These influence how CPUs drive the course (lines, drifting behavior, etc).

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position vector.
    0x0C: Fx32 — Point scale (used for weight or radius).
    0x10: Int16 — Drifting parameter (signed).
    0x12: UInt16 — Unknown (padding/flags).
    0x14: UInt32 — Unknown (engine-specific metadata).

    Gameplay Context
    ----------------
    - CPU behavior heavily depends on EPOI positions and the drifting
      parameter — altering these can change how tight/loose CPUs corner.
    - EPOI groups are linked via EPAT entries (below).

    Notes
    -----
    - The meaning of the 0x14 32-bit word is partially unknown in community
      docs; experiments suggest it can contain flags for AI behavior.
    """
    def __init__(self, data):
        super().__init__(data, 0x18)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.point_scale = [read_fx32(d, 0x0C) for d in self]
        self.drifting = [read_u16(d, 0x10) for d in self]
        self.unknown1 = [read_u16(d, 0x12) for d in self]
        self.unknown2 = [read_u32(d, 0x14) for d in self]


class EPAT(Section):
    """
    EPAT — EPOI grouping.

    Stride: 0x0C bytes per entry.

    Purpose
    -------
    Groups EPOI points into contiguous blocks and defines adjacency (next/prev
    groups) and section ordering. Used by CPU pathing code to navigate routes.

    Offsets (per-entry)
    -------------------
    0x00: UInt16 — Point start into EPOI array.
    0x02: UInt16 — Point length.
    0x04..0x06: Byte[3] — Next groups (0xFF unused).
    0x07..0x09: Byte[3] — Previous groups (0xFF unused).
    0x0A: Int16 — Section order.

    Gameplay Context
    ----------------
    - EPAT partitions EPOI arrays into logical AI routes; enabling multiple
      CPU strategies per segment.
    """
    def __init__(self, data):
        super().__init__(data, 0x0C)
        self.point_start = [read_u16(d, 0x00) for d in self]
        self.point_length = [read_u16(d, 0x02) for d in self]
        self.next_group = [
            [read_u8(d, i) for i in range(0x04, 0x07, 0x01)]
            for d in self
        ]
        self.prev_group = [
            [read_u8(d, i) for i in range(0x07, 0x0A, 0x01)]
            for d in self
        ]
        self.section_order = [read_s16(d, 0x0A) for d in self]


class MEPO(Section):
    """
    MEPO — Mini-game enemy points.

    Stride: 0x18 bytes per entry.

    Purpose
    -------
    Similar to EPOI but used by specific mini-games; describes where minigame
    entities spawn/move.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position.
    0x0C: Fx32 — Point scale.
    0x10: Int32 — Drifting (signed; larger type vs EPOI).
    0x14: UInt32 — Unknown.

    Gameplay Context
    ----------------
    - MEPO entries are used only in special mini-game contexts and may allow
      more variety (hence Int32 for drifting).
    """
    def __init__(self, data):
        super().__init__(data, 0x18)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.point_scale = [read_fx32(d, 0x0C) for d in self]
        self.drifting = [read_u32(d, 0x10) for d in self]
        self.unknown = [read_u32(d, 0x14) for d in self]


class MEPA(Section):
    """
    MEPA — MEPO grouping (for mini-games).

    Stride: 0x14 bytes per entry.

    Purpose
    -------
    Groups MEPO points into sequences. MEPA entries support up to 8 next and
    8 previous groups (Byte[8]) because mini-games may have richer topology.

    Offsets (per-entry)
    -------------------
    0x00: UInt16 — Point start index into MEPO array.
    0x02: UInt16 — Point length.
    0x04..0x0B: Byte[8] — Next group indices (0xFF unused).
    0x0C..0x13: Byte[8] — Previous group indices (0xFF unused).

    Gameplay Context
    ----------------
    - Use MEPA to create complex mini-game movement graphs (multiple branching).
    """
    def __init__(self, data):
        super().__init__(data, 0x14)
        self.point_start = [read_u16(d, 0x00) for d in self]
        self.point_length = [read_u16(d, 0x02) for d in self]
        self.next_group = [
            [read_u8(d, i) for i in range(0x04, 0x0C, 0x01)]
            for d in self
        ]
        self.prev_group = [
            [read_u8(d, i) for i in range(0x0C, 0x14, 0x01)]
            for d in self
        ]


class AREA(Section):
    """
    AREA — Camera/zone areas.

    Stride: 0x48 bytes per entry.

    Purpose
    -------
    Defines 3D regions used by the engine for camera selection and environmental
    triggers (sounds like waterfalls). Each area contains a center position,
    axes/length vectors (defining an oriented bounding box), and metadata such
    as area type and camera ID.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position vector (center).
    0x0C: Vector — Length vector (extent in X/Y/Z possibly).
    0x18: Vector — X-vector (orientation).
    0x24: Vector — Y-vector.
    0x30: Vector — Z-vector.
    0x3C: Int16 — Unknown.
    0x3E: Int16 — Unknown.
    0x40: Int16 — Unknown.
    0x42: Byte  — Unknown.
    0x43: Byte  — Camera ID (index into CAME).
    0x44: Byte  — Area type:
             - 0x00: Unknown
             - 0x01: Camera
             - 0x02: (unknown)
             - 0x03: (unknown)
             - 0x04: Waterfall sound area
    0x45: Int16 — Unknown.
    0x47: Byte  — Unknown.

    Gameplay Context
    ----------------
    - AREA sections usually determine which camera the engine should switch to
      when the player is inside the region.
    - Camera ID links to CAME entries; Area type allows special behavior (e.g.,
      waterfall sound triggers).
    - Good for editors to preview camera transitions.

    Notes
    -----
    - Several fields are still undocumented; their meaning varies by track.
    """
    def __init__(self, data):
        super().__init__(data, 0x48)
        self.position = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.length_vec = [read_vector_3d_fx32(d, 0x0C) for d in self]
        self.x_vec = [read_vector_3d_fx32(d, 0x18) for d in self]
        self.y_vec = [read_vector_3d_fx32(d, 0x24) for d in self]
        self.z_vec = [read_vector_3d_fx32(d, 0x30) for d in self]
        self.unknown1 = [read_u16(d, 0x3C) for d in self]
        self.unknown2 = [read_u16(d, 0x3E) for d in self]
        self.unknown3 = [read_u16(d, 0x40) for d in self]
        self.unknown4 = [read_u8(d, 0x42) for d in self]
        self.camera_id = [read_u8(d, 0x43) for d in self]
        self.area_type = None  # TODO: parse as Byte at 0x44 if desired
        self.unknown5 = [read_s16(d, 0x45) for d in self]
        self.unknown6 = [read_u8(d, 0x47) for d in self]


class CAME(Section):
    """
    CAME — Camera definitions.

    Stride: 0x4C bytes per entry.

    Purpose
    -------
    Defines camera motions and static camera positions used in cutscenes,
    intros, and dynamic in-game camera triggers. Camera entries are often linked
    by AREA zones or object routes.

    Offsets (per-entry)
    -------------------
    0x00: Vector — 3D position vector 1 (primary).
    0x0C: Vector — 3D rotation vector.
    0x18: Vector — 3D position vector 2 (secondary/look-at).
    0x24: Vector — 3D position vector 3 (tertiary/intermediate).
    0x30: Int16  — FOV begin (field-of-view start).
    0x32: Fx16  — FOV begin sine (precomputed).
    0x34: Fx16  — FOV begin cosine.
    0x36: Int16  — FOV end.
    0x38: Fx16  — FOV end sine.
    0x3A: Fx16  — FOV end cosine.
    0x3C: UInt16 — Camera zoom.
    0x3E: UInt16 — Camera type (see camera type table below).
    0x40: UInt16 — Linked route (0xFFFF if none).
    0x42: UInt16 — Route speed.
    0x44: UInt16 — Point speed.
    0x46: UInt16 — Camera duration (in 1/60s units).
    0x48: UInt16 — Next camera (0xFFFF if last).
    0x4A: Byte   — Intro pan first camera indicator:
    0x00 = none, 0x01 = top screen, 0x02 = bottom screen.
    0x4B: Byte   — Unknown (1 if camera type == 5 in some tracks).

    Camera Type (common):
    0x00: After race camera
    0x01: Unknown (with route)
    0x02: Unknown
    0x03: Intro camera (top screen)
    0x04: Intro camera (bottom screen)
    0x05: Unknown
    0x06: Unknown
    0x07: Battle mode camera
    0x08: Mission finish camera

    Gameplay Context
    ----------------
    - CAME entries drive cinematic camera movement during intros, cutscenes,
      and special camera modes.
    - Linked route + point speed define how the camera follows a PATH.
    - Camera duration uses 1/60th-second units — helpful for exact timing.

    Notes & Community Tips
    ----------------------
    - Many of these fields are precomputed (sine/cosine) for faster in-engine
      interpolation. Editors can either preserve or recompute them.
    - The "next camera" field allows building camera chains for sequences.
    - When building camera editors, expose both positions and FOV values to
      enable previewing transitions correctly.
    """
    def __init__(self, data):
        super().__init__(data, 0x4C)
        self.position1 = [read_vector_3d_fx32(d, 0x00) for d in self]
        self.rot_vec = [read_vector_3d_fx32(d, 0x0C) for d in self]
        self.position2 = [read_vector_3d_fx32(d, 0x18) for d in self]
        self.position3 = [read_vector_3d_fx32(d, 0x24) for d in self]
        self.fov_begin = [read_u16(d, 0x30) for d in self]
        self.fov_begin_sine = [read_fx16(d, 0x32) for d in self]
        self.fov_begin_cosine = [read_fx16(d, 0x34) for d in self]
        self.fov_end = [read_u16(d, 0x36) for d in self]
        self.fov_end_sine = [read_fx16(d, 0x38) for d in self]
        self.fov_end_cosine = [read_fx16(d, 0x3A) for d in self]
        self.camera_zoom = [read_u16(d, 0x3C) for d in self]
        self.camera_type = None  # TODO: parse if you want the raw value at 0x3E
        self.linked_route = [read_u16(d, 0x40) for d in self]
        self.route_speed = [read_u16(d, 0x42) for d in self]
        self.point_speed = [read_u16(d, 0x44) for d in self]
        self.camera_duration = [read_u16(d, 0x46) for d in self]
        self.next_camera = [read_u16(d, 0x48) for d in self]
        self.intro_pan_first_camera_indicator = [read_u8(d, 0x4A) for d in self]
        self.unknown = [read_u8(d, 0x4B) for d in self]


class NKM:
    """
    NKM — Mario Kart DS Course Map parser.

    Purpose
    -------
    Top-level container that parses the NKM file header and initializes objects
    representing each section (OBJI, PATH, POIT, STAG, KTPS, ... , CAME).

    Usage Example
    ----------------
    >>> nkm = NKM.from_file("my_course.nkm")
    >>> print(len(nkm._OBJI))         # number of object instances
    >>> print(nkm._STAG.amt_of_laps)  # global lap count for the stage

    Implementation Details
    ----------------------
    * The typical header length is 0x4C and contains offsets (UInt32) to 17
      canonical sections in standard order. Offsets are relative to the end of
      the header (H), so they are added to `_header_offset` to compute absolute
      positions inside the file.
    * This parser assumes the "typical" header layout. If an NKM contains
      additional special sections (like NKMI) or a different header length,
      additional handling will be required.
    * Parsing is defensive but assumes the file is well-formed; for robust
      tools consider adding bounds checks when slicing `self._data[...]`.

    Known Limitations
    -----------------
    - Some fields (GXRgb, camera_type, some unknown bytes) are left as None or
      unknown placeholders. Implementing GXRgb and Fx16/Fx32 conversions will
      enable richer output.
    - The code currently uses `PATH.has_loop = read_u8(...) != 1` which inverts
      the canonical meaning (spec: 1 means loop). Consider normalizing this if
      you rely on literal interpretation elsewhere.

    See Also
    --------
    The official NKM spec (community wiki) for exhaustive explanation of flags
    and object-specific settings.
    """
    _header_offset = 0x4C

    def __init__(self, data):
        self._data = data

        self._h = 0x4C
        # Header contains offsets relative to header_end; add header_offset
        self._OBJI_offset = read_u32(data, 0x08) + NKM._header_offset
        self._PATH_offset = read_u32(data, 0x0C) + NKM._header_offset
        self._POIT_offset = read_u32(data, 0x10) + NKM._header_offset
        self._STAG_offset = read_u32(data, 0x14) + NKM._header_offset
        self._KTPS_offset = read_u32(data, 0x18) + NKM._header_offset
        self._KTPJ_offset = read_u32(data, 0x1C) + NKM._header_offset
        self._KTP2_offset = read_u32(data, 0x20) + NKM._header_offset
        self._KTPC_offset = read_u32(data, 0x24) + NKM._header_offset
        self._KTPM_offset = read_u32(data, 0x28) + NKM._header_offset
        self._CPOI_offset = read_u32(data, 0x2C) + NKM._header_offset
        self._CPAT_offset = read_u32(data, 0x30) + NKM._header_offset
        self._IPOI_offset = read_u32(data, 0x34) + NKM._header_offset
        self._IPAT_offset = read_u32(data, 0x38) + NKM._header_offset
        self._EPOI_offset = read_u32(data, 0x3C) + NKM._header_offset
        self._EPAT_offset = read_u32(data, 0x40) + NKM._header_offset
        self._AREA_offset = read_u32(data, 0x44) + NKM._header_offset
        self._CAME_offset = read_u32(data, 0x48) + NKM._header_offset

        # Instantiate section objects. Slicing uses computed offsets.
        self._OBJI = OBJI(self._data[self._OBJI_offset:self._PATH_offset])
        self._PATH = PATH(self._data[self._PATH_offset:self._POIT_offset])
        self._POIT = POIT(self._data[self._POIT_offset:self._STAG_offset])
        self._STAG = STAG(self._data[self._STAG_offset:self._KTPS_offset])
        self._KTPS = KTPS(self._data[self._KTPS_offset:self._KTPJ_offset])
        self._KTPJ = KTPJ(self._data[self._KTPJ_offset:self._KTP2_offset])
        self._KTP2 = KTP2(self._data[self._KTP2_offset:self._KTPC_offset])
        self._KTPC = KTPC(self._data[self._KTPC_offset:self._KTPM_offset])
        self._KTPM = KTPM(self._data[self._KTPM_offset:self._CPOI_offset])
        self._CPOI = CPOI(self._data[self._CPOI_offset:self._CPAT_offset])
        self._CPAT = CPAT(self._data[self._CPAT_offset:self._IPOI_offset])
        self._IPOI = IPOI(self._data[self._IPOI_offset:self._IPAT_offset])
        self._IPAT = IPAT(self._data[self._IPAT_offset:self._EPOI_offset])
        self._EPOI = EPOI(self._data[self._EPOI_offset:self._EPAT_offset])
        self._EPAT = EPAT(self._data[self._EPAT_offset:self._AREA_offset])
        self._AREA = AREA(self._data[self._AREA_offset:self._CAME_offset])
        self._CAME = CAME(self._data[self._CAME_offset:])

    @classmethod
    def from_file(cls, path: str, **kwargs):
        """
        Load an NKM file from disk and parse its sections.

        Args:
            path (str): Path to the `.nkm` file. Defaults to DEFAULT_NKM_PATH.

        Returns:
            NKM: The parsed NKM object.
        """
        import os
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"NKM file not found at {path}")
        with open(path, 'rb') as f:
            return cls(f.read(), **kwargs)
