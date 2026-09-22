/* bb_config.h: fixed sizes. Nothing in this library allocates after init.
 *
 * Every bound here is checked when a target bundle is parsed; a bundle that needs more is
 * rejected with BB_ERR_TOO_BIG rather than truncated.
 */
#ifndef BB_CONFIG_H
#define BB_CONFIG_H

#define BB_K              8    /* slot rows per scene state */
#define BB_W             14    /* float32 columns per slot row */

#define BB_MAX_REFS       8    /* reference blobs one step may line up on */
#define BB_MAX_FEAT      (2 * BB_MAX_REFS)   /* rows of the feature Jacobian */
#define BB_MAX_BROW       8    /* published columns read for the in-hand fit */
#define BB_MAX_BDOF       3    /* in-hand unknowns solved for */

#define BB_MAX_CMDS      12    /* 50 Hz writes one vision tick may issue */
#define BB_MATCH_MAX     16    /* frames back the actual state may be matched to */

#define BB_NJ             5    /* arm joints the kinematics solves for */
#define BB_NJOINT         6    /* joints on the bus: the five plus the gripper */
#define BB_NLINK          5    /* moving links between the base and the tool */

/* Slot-row column offsets, the PLACEHOLDER_SLOT_SCHEMA layout. */
#define BB_COL_MASS       0
#define BB_COL_CENTRE     1
#define BB_COL_MOMENTS    3
#define BB_COL_COLOUR     6
#define BB_COL_SPREAD     9
#define BB_COL_BORDER    10
#define BB_COL_AGE       11
#define BB_COL_ORIENT4   12

/* C99 does not define M_PI; the controller carries its own. */
#define BB_PI 3.14159265358979323846

/* Return codes. */
#define BB_OK             0
#define BB_ERR_MAGIC     -1
#define BB_ERR_VERSION   -2
#define BB_ERR_CRC       -3
#define BB_ERR_TRUNCATED -4
#define BB_ERR_TOO_BIG   -5
#define BB_ERR_LAYOUT    -6

#endif /* BB_CONFIG_H */
