#include "mcu_periph/sys_time.h"
#include "mav_course_exercise.h"
#include <stdio.h>
#include "paparazzi.h"
#include "modules/core/abi.h"
#include "modules/computer_vision/opticflow/opticflow_calculator.h"

static abi_event opticflow_ev;

// Optical Flow Callback: Logt alleen de ruwe waardes
static void opticflow_cb(uint8_t sender_id, int16_t motionX, int16_t motionY, 
                          int16_t velocityX, int16_t velocityY, 
                          int32_t quality, float div_size) {
    // Print direct de Optical Flow waarden bij ELKE update
    printf("[Optical Flow] MotionX: %d, MotionY: %d, VelocityX: %d, VelocityY: %d, Quality: %d, DivSize: %.2f\n",
           motionX, motionY, velocityX, velocityY, quality, div_size);
}

// Initialisatie van Optical Flow
void mav_course_exercise_init(void) {
    printf("[MAV Course Exercise] Module initialized!\n");
    AbiBindMsgOPTICAL_FLOW(ABI_BROADCAST, &opticflow_ev, opticflow_cb);
}

// Periodieke functie: Geen extra logs, alleen Optical Flow wordt automatisch geprint via de callback
void mav_course_exercise_periodic(void) {
    // Hier niets printen, Optical Flow wordt al in de callback gelogd
}
