#ifndef INCLUDED_tracer_input_fields_h_
#define INCLUDED_tracer_input_fields_h_

typedef struct {
    const char *x;
    const char *y;
    const char *z;
    const char *vx;
    const char *vy;
    const char *vz;
    const char *curlv;
    const char *divv;
    const char *bmag;
    const char *bx;
    const char *by;
    const char *bz;
    const char *temperature;
    const char *density;
    const char *time;
    const char *redshift;
    const char *volume;
    const char *mass;
    const char *lturb;
    const char *lconv;
    const char *dconv;
    const char *vconv;
    const char *tconv;
    const char *mach;
    const char *prestemp;
    const char *presden;
    const char *pre_density_cgs;
    const char *upstream_speed_cgs;
    const char *shock_side_code;
    const char *dsa_trigger;
    const char *dsa_mach;
    const char *dsa_pre_density;
    const char *dsa_kinetic_energy_flux_cgs;
    const char *particle_id;
    const char *mask;
} TracerInputFieldNames;

/* Centralized HDF5 dataset names for tracer input files. */
static const TracerInputFieldNames kTracerInputFields = {
    "x_coordinate",   
    "y_coordinate",
    "z_coordinate",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "curl_v",      ///  velocity field derivatie, rotation v, 
    "div_v",       ///  velocity field derivatie, divergence v, 
    "B",
    "B_x",
    "B_y",
    "B_z",
    "temperature",
    "density",
    "time",
    "Redshift",
    "volume",      // physical volume of tracer (optional)
    "M_tracer",    // physical mass of tracer in M_sun (optional)
    "dx_phys",     // physical (not comoving) scale of turbulence (scale at which curl_v and div_v are measured)
    "lUnit",       //  length unit of cosmological sim
    "dUnit",       //  density unit of cosmological sim
    "vUnit",       //  velocity unit of cosmological sim
    "tUnit",       //  time unit of cosmological sim
    "mach",       
    
    // DSA related fields (optional)
    "prestemp",        
    "presden",
    "pre_density_cgs",
    "upstream_speed_cgs",
    "shock_side_code",
    "dsa_trigger",
    "dsa_mach",
    "dsa_pre_density",
    "dsa_kinetic_energy_flux_cgs",

    "particle_ID",  // tracer particle ID. should be common across snapshots

    "mask_flag" 
};

#define DATASET_NAME_X kTracerInputFields.x
#define DATASET_NAME_Y kTracerInputFields.y
#define DATASET_NAME_Z kTracerInputFields.z
#define DATASET_NAME_VX kTracerInputFields.vx
#define DATASET_NAME_VY kTracerInputFields.vy
#define DATASET_NAME_VZ kTracerInputFields.vz
#define DATASET_NAME_CURLV kTracerInputFields.curlv
#define DATASET_NAME_DIVV kTracerInputFields.divv
#define DATASET_NAME_B kTracerInputFields.bmag
#define DATASET_NAME_Bx kTracerInputFields.bx
#define DATASET_NAME_By kTracerInputFields.by
#define DATASET_NAME_Bz kTracerInputFields.bz
#define DATASET_NAME_TEMP kTracerInputFields.temperature
#define DATASET_NAME_DENSITY kTracerInputFields.density
#define DATASET_NAME_TIME kTracerInputFields.time
#define DATASET_NAME_REDSHIFT kTracerInputFields.redshift
#define DATASET_NAME_VOLUME kTracerInputFields.volume
#define DATASET_NAME_MASS kTracerInputFields.mass
#define DATASET_NAME_Lturb kTracerInputFields.lturb
#define DATASET_NAME_LCONV kTracerInputFields.lconv
#define DATASET_NAME_DCONV kTracerInputFields.dconv
#define DATASET_NAME_VCONV kTracerInputFields.vconv
#define DATASET_NAME_TCONV kTracerInputFields.tconv
#define DATASET_NAME_MACH kTracerInputFields.mach
#define DATASET_NAME_PRETEMP kTracerInputFields.prestemp
#define DATASET_NAME_PRESDEN kTracerInputFields.presden
#define DATASET_NAME_PRE_DENSITY_CGS kTracerInputFields.pre_density_cgs
#define DATASET_NAME_UPSTREAM_SPEED_CGS kTracerInputFields.upstream_speed_cgs
#define DATASET_NAME_SHOCK_SIDE_CODE kTracerInputFields.shock_side_code
#define DATASET_NAME_DSA_TRIGGER kTracerInputFields.dsa_trigger
#define DATASET_NAME_DSA_MACH kTracerInputFields.dsa_mach
#define DATASET_NAME_DSA_PRE_DENSITY kTracerInputFields.dsa_pre_density
#define DATASET_NAME_DSA_KINETIC_ENERGY_FLUX_CGS kTracerInputFields.dsa_kinetic_energy_flux_cgs
#define DATASET_NAME_ID kTracerInputFields.particle_id
#define DATASET_NAME_MASK kTracerInputFields.mask

#endif
