typedef struct driver_t
{
    sfx_emitter_t soundEmitter;
    u32 field44;
    u32 flags;
    u32 flags2;
    VecFx32 direction;
    VecFx32 drivingDirection;
    VecFx32 velocity;
    u16 id;
    InputUnitId inputId;
    u32 field7C;
    VecFx32 position;
    VecFx32 lastPosition;
    VecFx32 kartTiresPosition;
    // members ommitted ...
    VecFx32 thunderScale;
    fx32 dossunYScale;
    mobj_inst_t* mobjHitList[2];
    u16 mobjHitSfxTimeout[2];
    BOOL mobjHitEmittedSfx[2];
    mobj_inst_t* smashDossun;
    driver_field450_t field450;
    fx32 field4BC;
    u32 colFlagsMap2DShadow;
    u32 jumpPadSpeed;
    fx32 field4C8;
    // members ommitted ...
    u16 field504;
    u16 field506;
    VecFx32* field508;
    quaternion_t* field50C;
    VecFx32* field510;
    driver_net_state_t* netState;
    sfx_emitter_ex_params_t field518;
    void* field534;
    driver_timers_t timers;
    charkart_t* charKart;
    // members ommitted ...
} driver_t;