/*
 * Minimal UAV Module - MAV Course Exercise
 * Prints Optical Flow valuess
 */

 #include "mcu_periph/sys_time.h"
 #include "mav_course_exercise.h"  // Eigen header file
 #include <stdio.h>                // Standaard C-library voor printf
 #include "paparazzi.h"            // Paparazzi basisfuncties
 #include "modules/core/abi.h"
 // ABI functies voor communicatie
 #include "modules/computer_vision/opticflow/opticflow_calculator.h"  // OPTICAL FLOW MODULE
 
 #define PRINT_INTERVAL 1000  // 1000 ms = 1 seconde
 
 static uint32_t last_print_time = 0; // Houdt de laatste printtijd bij
 
 // Definieer de Optical Flow event
 static abi_event opticflow_ev;
 
 // ** CALLBACK FUNCTIE: Optical Flow Data Printen **
 static void opticflow_cb(uint8_t sender_id, int16_t motionX, int16_t motionY, 
                          int16_t velocityX, int16_t velocityY, 
                          int32_t quality, float div_size) {
     printf("[Optical Flow] MotionX: %d, MotionY: %d, VelocityX: %d, VelocityY: %d, Quality: %d, DivSize: %.2f\n",
            motionX, motionY, velocityX, velocityY, quality, div_size);
 }
 
 // ** INITIALISATIE **
 void mav_course_exercise_init(void) {
     printf("[MAV Course Exercise] Module initialized!\n");
 
     // ** Optical Flow Callback Binden **
     AbiBindMsgOPTICAL_FLOW(ABI_BROADCAST, &opticflow_ev, opticflow_cb);
 }
 
 // ** PERIODIEKE FUNCTIE **
 void mav_course_exercise_periodic(void) {
     uint32_t now = get_sys_time_msec(); // Huidige tijd ophalen
     if (now - last_print_time > PRINT_INTERVAL) {
         printf("[MAV Course Exercise] Running! Time: %u ms\n", now);
         last_print_time = now;
     }
 }
 