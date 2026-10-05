#ifdef __cplusplus
extern "C" {
#endif

int read_param_file(int rank, const char *filename);
int read_param_file_noMPI(const char *filename);
int finalize_dpp_mode_config(int legacy_param_seen);

#ifdef __cplusplus
}
#endif
