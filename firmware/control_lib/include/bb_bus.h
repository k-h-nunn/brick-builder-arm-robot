/* bb_bus.h: the servo bus, behind two function pointers.
 *
 * The controller writes joint targets and reads back positions, speeds and loads, and knows
 * nothing else about how that happens. Three implementations plug in here:
 *
 *   - none at all (`bb_bus` left null): the plant is simulated on the host and the readings
 *     arrive over the wire as BUS_READINGS. This is what processor-in-the-loop runs.
 *   - the half-duplex STS3215 driver on the microcontroller: it fills in exactly these two
 *     entry points and nothing more.
 *   - a stub that echoes the last write back, for timing a tick with no hardware attached.
 *
 * `q` and the readings are in radians on the output shaft, speeds in radians per second, loads as
 * a signed fraction of the servo's own limit. Converting those to and from register counts is the
 * driver's business, not the controller's.
 */
#ifndef BB_BUS_H
#define BB_BUS_H

#include "bb_config.h"

typedef struct bb_bus bb_bus;

struct bb_bus {
    void *ctx;
    /* Write `n` joint targets with per-joint torque limits (1.0 = the servo's own limit).
     * Returns 0 on success, negative on a bus fault. */
    int (*write)(void *ctx, const double *q, const double *torque_limit, int n);
    /* Read `n` joints back. Any of the three output arrays may be null. */
    int (*read)(void *ctx, double *position, double *speed, double *load, int n);
};

#endif /* BB_BUS_H */
