/* p4_control: the run-time controller on the ESP32-P4.
 *
 * The same `firmware/control_lib` the host runs through ctypes and as `control_runner`, behind
 * the same `bb_session_run`, with two board-specific pieces:
 *
 *   - the transport: the wire protocol on UART0 (the board's USB-UART), 921600 baud, 8N1. The
 *     console is off so nothing else is ever written to that line.
 *   - the servo bus: `bb_session` hands every 50 Hz joint target to a queue; a task woken by a
 *     20 ms esp_timer writes one per period to the `bb_bus` seam. Until the STS3215 driver
 *     exists the seam is `bb_bus_loopback`. In processor-in-the-loop runs the plant is simulated
 *     on the host and the host's BUS_READINGS replace the loopback's before every SCENE_STATE.
 *
 * STATUS reports the controller's time per vision tick (mean and worst, `bb_tick_begin` plus
 * `bb_tick_end`) and the bus writer's counters: writes, targets dropped because the queue was
 * full, and the longest interval between two bus periods.
 */
#include <stdint.h>
#include <string.h>

#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "bb_bus_loopback.h"
#include "bb_session.h"

#define PROTO_UART      UART_NUM_0
#define PROTO_BAUD      921600
#define UART_RX_BYTES   16384
#define UART_TX_BYTES   8192

#define BUNDLE_CAP_PSRAM     (4u * 1024u * 1024u)
#define BUNDLE_CAP_INTERNAL  (448u * 1024u)
#define INTERNAL_HEADROOM    (48u * 1024u)

#define BUS_PERIOD_US   20000
#define BUS_QUEUE_DEPTH (2 * BB_MAX_CMDS)

#define SESSION_STACK   16384
#define BUS_STACK       4096

typedef struct {
    double q[BB_NJOINT];
    double torque[BB_NJOINT];
} bus_write_t;

static bb_session s_session;
static uint8_t *s_bundle;
static uint32_t s_bundle_cap;

static QueueHandle_t s_bus_queue;
static TaskHandle_t s_bus_task;
static bb_bus_loopback s_loopback;
static bb_bus s_device;      /* what the 50 Hz task writes to: the loopback, later the STS3215 */
static bb_bus s_paced;       /* what the session sees: writes queue, reads return the snapshot */
static portMUX_TYPE s_snapshot_lock = portMUX_INITIALIZER_UNLOCKED;
static double s_position[BB_NJOINT];
static double s_speed[BB_NJOINT];
static double s_load[BB_NJOINT];

/* -- transport: UART0 -------------------------------------------------------------------- */

static int uart_rd(void *ctx, uint8_t *buf, uint32_t n)
{
    size_t avail = 0;
    (void)ctx;
    uart_get_buffered_data_len(PROTO_UART, &avail);
    if (avail == 0) {
        /* Block for the first byte, then take whatever else has arrived. */
        return uart_read_bytes(PROTO_UART, buf, 1, portMAX_DELAY);
    }
    if (avail > n) {
        avail = n;
    }
    return uart_read_bytes(PROTO_UART, buf, (uint32_t)avail, 0);
}

static int uart_wr(void *ctx, const uint8_t *buf, uint32_t n)
{
    (void)ctx;
    return uart_write_bytes(PROTO_UART, (const char *)buf, n);
}

static uint64_t now_us(void *ctx)
{
    (void)ctx;
    return (uint64_t)esp_timer_get_time();
}

static void uart_start(void)
{
    const uart_config_t cfg = {
        .baud_rate = PROTO_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(PROTO_UART, UART_RX_BYTES, UART_TX_BYTES, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(PROTO_UART, &cfg));
    /* UART0 on its default pins, the ones the USB-UART bridge is wired to. */
    ESP_ERROR_CHECK(uart_set_pin(PROTO_UART, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

/* -- the servo bus, paced at 50 Hz --------------------------------------------------------- */

static int paced_write(void *ctx, const double *q, const double *torque_limit, int n)
{
    bus_write_t w;
    int i;
    (void)ctx;
    for (i = 0; i < BB_NJOINT; ++i) {
        w.q[i] = i < n ? q[i] : 0.0;
        w.torque[i] = (i < n && torque_limit != NULL) ? torque_limit[i] : 1.0;
    }
    if (xQueueSend(s_bus_queue, &w, 0) != pdTRUE) {
        /* Full: the oldest target is the stale one. */
        bus_write_t old;
        xQueueReceive(s_bus_queue, &old, 0);
        s_session.bus_dropped += 1u;
        xQueueSend(s_bus_queue, &w, 0);
    }
    return 0;
}

static int paced_read(void *ctx, double *position, double *speed, double *load, int n)
{
    int i;
    (void)ctx;
    if (n > BB_NJOINT) {
        n = BB_NJOINT;
    }
    taskENTER_CRITICAL(&s_snapshot_lock);
    for (i = 0; i < n; ++i) {
        if (position != NULL) {
            position[i] = s_position[i];
        }
        if (speed != NULL) {
            speed[i] = s_speed[i];
        }
        if (load != NULL) {
            load[i] = s_load[i];
        }
    }
    taskEXIT_CRITICAL(&s_snapshot_lock);
    return 0;
}

static void bus_tick(void *arg)
{
    (void)arg;
    xTaskNotifyGive(s_bus_task);
}

static void bus_task(void *arg)
{
    int64_t last = 0;
    (void)arg;
    for (;;) {
        bus_write_t w;
        double p[BB_NJOINT], v[BB_NJOINT], l[BB_NJOINT];
        int64_t now;
        int i;

        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        /* The bus counters in s_session are written here on core 0 without a lock, while
         * bb_session_init on core 1 clears the whole struct at the start of every session. An
         * update that races the clear can be lost or survive it. Only STATUS reads the
         * counters; nothing in the control path depends on them. */
        now = esp_timer_get_time();
        if (last != 0) {
            uint32_t gap = (uint32_t)(now - last);
            if (gap > s_session.bus_period_us_max) {
                s_session.bus_period_us_max = gap;
            }
        }
        last = now;
        if (xQueueReceive(s_bus_queue, &w, 0) == pdTRUE) {
            s_device.write(s_device.ctx, w.q, w.torque, BB_NJOINT);
            s_session.bus_writes += 1u;
        }
        s_device.read(s_device.ctx, p, v, l, BB_NJOINT);
        taskENTER_CRITICAL(&s_snapshot_lock);
        for (i = 0; i < BB_NJOINT; ++i) {
            s_position[i] = p[i];
            s_speed[i] = v[i];
            s_load[i] = l[i];
        }
        taskEXIT_CRITICAL(&s_snapshot_lock);
    }
}

static void bus_start(void)
{
    const esp_timer_create_args_t args = {
        .callback = bus_tick,
        .arg = NULL,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "bus50hz",
    };
    esp_timer_handle_t timer;

    s_bus_queue = xQueueCreate(BUS_QUEUE_DEPTH, sizeof(bus_write_t));
    bb_bus_loopback_init(&s_loopback, &s_device, BUS_PERIOD_US * 1e-6);
    s_paced.ctx = NULL;
    s_paced.write = paced_write;
    s_paced.read = paced_read;
    xTaskCreatePinnedToCore(bus_task, "bus", BUS_STACK, NULL, configMAX_PRIORITIES - 2,
                            &s_bus_task, 0);
    ESP_ERROR_CHECK(esp_timer_create(&args, &timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(timer, BUS_PERIOD_US));
}

/* -- the target bundle's buffer -------------------------------------------------------------- */

static void bundle_alloc(void)
{
    size_t largest;

    s_bundle = heap_caps_aligned_alloc(8, BUNDLE_CAP_PSRAM, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_bundle != NULL) {
        s_bundle_cap = BUNDLE_CAP_PSRAM;
        return;
    }
    largest = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    s_bundle_cap = BUNDLE_CAP_INTERNAL;
    if (largest < (size_t)s_bundle_cap + INTERNAL_HEADROOM) {
        s_bundle_cap = largest > INTERNAL_HEADROOM ? (uint32_t)(largest - INTERNAL_HEADROOM) : 0u;
        s_bundle_cap &= ~7u;
    }
    s_bundle = s_bundle_cap ? heap_caps_aligned_alloc(8, s_bundle_cap,
                                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
                            : NULL;
    if (s_bundle == NULL) {
        s_bundle_cap = 0u;
    }
}

/* -- the session ---------------------------------------------------------------------------- */

static void session_task(void *arg)
{
    bb_transport io;
    (void)arg;
    io.ctx = NULL;
    io.read = uart_rd;
    io.write = uart_wr;
    io.now_us = now_us;
    for (;;) {
        /* One session per host connection; a RESET or a read error starts the next one, which
         * greets the host again. HELLO carries the bundle capacity, 0 when no buffer could be
         * had, and every bundle is then refused. */
        bb_session_init(&s_session, &io, s_bundle, s_bundle_cap, &s_paced);
        bb_session_run(&s_session);
    }
}

void app_main(void)
{
    bundle_alloc();
    uart_start();
    bus_start();
    xTaskCreatePinnedToCore(session_task, "session", SESSION_STACK, NULL,
                            configMAX_PRIORITIES - 4, NULL, 1);
}
