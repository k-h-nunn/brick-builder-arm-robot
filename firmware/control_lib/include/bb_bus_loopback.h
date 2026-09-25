/* bb_bus_loopback.h: a servo bus with nothing on it.
 *
 * `write` stores the targets; `read` reports them back as the positions, the change since the
 * previous write over `dt` as the speeds, and zero loads. It stands behind the `bb_bus` seam
 * until the STS3215 driver does, so the write path, the 50 Hz pacing and the tick timing can be
 * exercised with no arm attached.
 */
#ifndef BB_BUS_LOOPBACK_H
#define BB_BUS_LOOPBACK_H

#include <stdint.h>
#include "bb_bus.h"
#include "bb_config.h"

typedef struct {
    double dt;                      /* seconds between writes, for the reported speed */
    double position[BB_NJOINT];
    double previous[BB_NJOINT];
    double torque_limit[BB_NJOINT];
    uint32_t writes;
    uint32_t reads;
} bb_bus_loopback;

/* Fill `bus` so its two entry points act on `lb`. */
void bb_bus_loopback_init(bb_bus_loopback *lb, bb_bus *bus, double dt);

#endif /* BB_BUS_LOOPBACK_H */
