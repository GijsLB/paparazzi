#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>

#define NAV_C
#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE FALSE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static int32_t color_count = 1; // Pretend we always see a small object


static abi_event horizon_detection_ev;
static void horizon_detection_cb(uint8_t sender_id, uint8_t *horizon, uint8_t length) {
    // Instead of using the horizon data, we force a constant detection.
    (void)sender_id;
    (void)horizon;
    (void)length;

    // Fake detection output
    color_count = 1;  // Pretend we always see a small object
}

void orange_avoider_init(void) {
    AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &horizon_detection_ev, horizon_detection_cb);
}
// Placeholder function until guidance logic is fixed
void guidance_turn_right() {
    // Dummy function: Pretend the drone turns right
}
void guidance_go_forward() {
    // Dummy function: Pretend the drone moves forward
}


void orange_avoider_periodic(void) {
    // Only run if in flight
    if (!autopilot_in_flight()) {
        return;
    }

    // Dummy response: Always assume a small obstacle
    int32_t color_count_threshold = 1;

    VERBOSE_PRINT("Color_count: %d  threshold: %d\n", color_count, color_count_threshold);

    if (color_count >= color_count_threshold) {
        guidance_turn_right();
    } else {
        guidance_go_forward();
    }
}

