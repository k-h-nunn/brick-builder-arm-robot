#include "bb_bus_loopback.h"

static int lb_write(void *ctx, const double *q, const double *torque_limit, int n)
{
    bb_bus_loopback *lb = (bb_bus_loopback *)ctx;
    int i;
    if (n > BB_NJOINT) {
        n = BB_NJOINT;
    }
    for (i = 0; i < n; ++i) {
        lb->previous[i] = lb->position[i];
        lb->position[i] = q[i];
        lb->torque_limit[i] = torque_limit != 0 ? torque_limit[i] : 1.0;
    }
    lb->writes += 1u;
    return 0;
}

static int lb_read(void *ctx, double *position, double *speed, double *load, int n)
{
    bb_bus_loopback *lb = (bb_bus_loopback *)ctx;
    int i;
    if (n > BB_NJOINT) {
        n = BB_NJOINT;
    }
    for (i = 0; i < n; ++i) {
        if (position != 0) {
            position[i] = lb->position[i];
        }
        if (speed != 0) {
            speed[i] = lb->dt > 0.0 ? (lb->position[i] - lb->previous[i]) / lb->dt : 0.0;
        }
        if (load != 0) {
            load[i] = 0.0;
        }
    }
    lb->reads += 1u;
    return 0;
}

void bb_bus_loopback_init(bb_bus_loopback *lb, bb_bus *bus, double dt)
{
    int i;
    lb->dt = dt;
    for (i = 0; i < BB_NJOINT; ++i) {
        lb->position[i] = 0.0;
        lb->previous[i] = 0.0;
        lb->torque_limit[i] = 1.0;
    }
    lb->writes = 0u;
    lb->reads = 0u;
    bus->ctx = lb;
    bus->write = lb_write;
    bus->read = lb_read;
}
