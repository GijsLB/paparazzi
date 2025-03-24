/**
 * File: opticalflow_avoider.c
 * Optical Flow Avoider met vector callbacks voor obstakel ontwijking.
 */

 #include "modules/opticalflow_avoider/opticalflow_avoider.h"
 #include "firmwares/rotorcraft/navigation.h"
 #include "modules/core/abi.h"
 #include "state.h"
 #include "math/pprz_algebra_float.h"
 #include "stdlib.h"
 #include "time.h"
 #include <stdio.h>
 #include <math.h>
 
 #define NAV_C
 #include "generated/flight_plan.h"
 
 // ======== Debugging macro ========
 #define OF_VERBOSE TRUE
 #if OF_VERBOSE
 #define OF_PRINT(fmt, ...) fprintf(stderr, "[OF Avoider->%s()] " fmt, __FUNCTION__, ##__VA_ARGS__)
 #else
 #define OF_PRINT(...)
 #endif
 
 // ======== State Machine ========
 enum of_state_t {
     OF_STATE_WAIT_FORWARD,
     OF_STATE_CHECK_OBSTACLE,
     OF_STATE_ROTATING
 };
 
 // ======== Parameters ========
 static float OF_THRESHOLD = 35.f;
 static float FORWARD_DIST = 1.0f;
 static float TURN_ANGLE   = 90.0f;
 float obstacle_threshold = 0;
 
 // ======== Global vars ========
 static enum of_state_t of_state = OF_STATE_WAIT_FORWARD;
 static int32_t motion_x = 0;
 static int32_t motion_y = 0;
 static double motion_magnitude = 0.0; // Maak het globaal
 static int32_t opticflow_vx = 0;
 static int32_t opticflow_vy = 0;
 static float last_state_start = 0.f;
 
 // ======== Forward-declaraties ========

//  static void of_vector_callback(uint8_t sender_id, uint16_t count, struct flow_t *vectors);
 static void of_set_state(enum of_state_t new_state);
 static uint8_t of_increase_heading(float delta_deg);
 static uint8_t of_move_waypoint_forward(uint8_t wp, float dist);
 static uint8_t of_calc_forwards(struct EnuCoor_i *coor, float dist);
 static uint8_t of_move_waypoint(uint8_t wp, struct EnuCoor_i *coor);

 static void of_vector_callback(uint8_t sender_id,
                uint8_t count,
                int32_t *flow_xy);



//  static void of_vector_callback(uint8_t sender_id,
//     uint8_t count,
//     int32_t flow_x0, int32_t flow_y0,
//     int32_t flow_x1, int32_t flow_y1,
//     int32_t flow_x2, int32_t flow_y2,
//     int32_t flow_x3, int32_t flow_y3,
//     int32_t flow_x4, int32_t flow_y4,
//     int32_t flow_x5, int32_t flow_y5,
//     int32_t flow_x6, int32_t flow_y6,
//     int32_t flow_x7, int32_t flow_y7,
//     int32_t flow_x8, int32_t flow_y8,
//     int32_t flow_x9, int32_t flow_y9);
 

 static abi_event of_vec_ev;

 /**
  * opticalflow_avoider_init
  */
 void opticalflow_avoider_init(void) {
     OF_PRINT("Init opticalflow avoider\n");
 

 
     AbiBindMsgOPTICAL_FLOW_VECTORS(OPTICAL_FLOW_CALCULATOR_ID,
                                    &of_vec_ev,
                                    of_vector_callback);
     
     OF_PRINT("OpticalFlow ABI bound, waiting for messages...\n");
 
     srand(time(NULL));
     of_set_state(OF_STATE_WAIT_FORWARD);
 }
 
 /**
  * opticalflow_avoider_periodic
  */
 void opticalflow_avoider_periodic(void) {
     if (!autopilot_in_flight() || autopilot_get_mode() != AP_MODE_NAV) {
         return;
     }
 
     float now_s = get_sys_time_float();
     float dt_s  = now_s - last_state_start;
     float psi_deg = DegOfRad(stateGetNedToBodyEulers_f()->psi);
     float nav_deg = DegOfRad(nav.heading);
 
     switch (of_state) {
         case OF_STATE_WAIT_FORWARD:
             if (dt_s > 1.0f) {
                 of_set_state(OF_STATE_CHECK_OBSTACLE);
             }
             break;
 
         case OF_STATE_CHECK_OBSTACLE:
            OF_PRINT("Checking obstacle: magnitude=%.2f threshold=%.1f\n", motion_magnitude, OF_THRESHOLD);
            if (fabsf(motion_magnitude) > OF_THRESHOLD) {
                OF_PRINT("Flow te hoog, draai 20 graden links\n");
                of_increase_heading(+90.0f);
                of_set_state(OF_STATE_ROTATING);
            } else {
                // Flow is onder drempel: ga rechtdoor
                of_move_waypoint_forward(WP_GOAL, FORWARD_DIST);
                of_set_state(OF_STATE_WAIT_FORWARD);
            }
            
            
            
            //  if (motion_magnitude < -OF_THRESHOLD) {
            //      OF_PRINT("Obstacle on right => turn left by %.1f deg\n", TURN_ANGLE);
            //      of_increase_heading(+TURN_ANGLE);
            //      of_set_state(OF_STATE_ROTATING);
            //  } else if (motion_magnitude > OF_THRESHOLD) {
            //      OF_PRINT("Obstacle on left => turn right by %.1f deg\n", TURN_ANGLE);
            //      of_increase_heading(-TURN_ANGLE);
            //      of_set_state(OF_STATE_ROTATING);
            //  } else {
            //      OF_PRINT("No obstacle => move WP forward=%.1f m\n", FORWARD_DIST);
            //      of_move_waypoint_forward(WP_GOAL, FORWARD_DIST);
            //      of_set_state(OF_STATE_WAIT_FORWARD);
            //  }
             break;
 
         case OF_STATE_ROTATING: {
             float current_psi = stateGetNedToBodyEulers_f()->psi;
             float diff = current_psi - nav.heading;
             FLOAT_ANGLE_NORMALIZE(diff);
 
             if (fabsf(diff) < RadOfDeg(5.f)) {
                 OF_PRINT("Done rotating => forward!\n");
                 of_move_waypoint_forward(WP_GOAL, FORWARD_DIST);
                 of_set_state(OF_STATE_WAIT_FORWARD);
             }
 
             if (dt_s > 8.0f) {
                 OF_PRINT("WARNING: rotating took too long => proceed anyway\n");
                 of_move_waypoint_forward(WP_GOAL, FORWARD_DIST);
                 of_set_state(OF_STATE_WAIT_FORWARD);
             }
         } break;
     }
 }
 

static void of_vector_callback(uint8_t sender_id,
    uint8_t count,
    int32_t *flow_xy)
{
fprintf(stderr, "[OF] cb: got count=%d from sender_id=%d\n", count, sender_id);

int64_t sum_fx = 0;  // 64-bit om overflows te vermijden als je count > ~32k
int64_t sum_fy = 0; 
// double motion_magnitude = 0.0; // Declareer motion_magnitude correct
// Loop over de x,y paren
for (int i = 0; i < count; i++) {
// x staat in flow_xy[2*i], y in flow_xy[2*i + 1]
int32_t fx = flow_xy[2*i];
int32_t fy = flow_xy[2*i + 1];
sum_fx += fx;
sum_fy += fy;

fprintf(stderr, "   i=%d => flow_x=%ld flow_y=%ld\n",
i, (long)fx, (long)fy);
}

if (count > 0) {
    motion_x = sum_fx / count;
    motion_y = sum_fy / count;
    motion_magnitude = sqrt((double)(motion_x * motion_x) + (double)(motion_y * motion_y)); // Euclidische norm
  } else {
    motion_x = 0;
    motion_y = 0;
    motion_magnitude = 0;
  }


  if (motion_magnitude > 100.0) {
    OF_PRINT("Flow magnitude too high (%.2f), discarding measurement\n", motion_magnitude);
    motion_x = 0;
    motion_y = 0;
    motion_magnitude = 0;
  }


  // Debug
  fprintf(stderr, "[OF] average fx = %ld, fy = %ld, magnitude = %.2f\n", 
    (long)motion_x, (long)motion_y, motion_magnitude);

}



 
 /**
  * Verander de state machine
  */
 static void of_set_state(enum of_state_t new_state) {
     of_state = new_state;
     last_state_start = get_sys_time_float();
     OF_PRINT("Switch state => %d\n", (int)new_state);
 }
 
 /**
  * Verhoog heading met delta
  */
 static uint8_t of_increase_heading(float delta_deg) {
     float curr_psi = stateGetNedToBodyEulers_f()->psi;
     float new_hdg = curr_psi + RadOfDeg(delta_deg);
     FLOAT_ANGLE_NORMALIZE(new_hdg);
     nav.heading = new_hdg;
     OF_PRINT("nav.heading=%.1f deg (current psi=%.1f deg, delta=%.1f deg)\n",
             DegOfRad(new_hdg), DegOfRad(curr_psi), delta_deg);
     return 0;
 }
 
 /**
  * Verplaats waypoint naar voren
  */
 static uint8_t of_move_waypoint_forward(uint8_t wp, float dist) {
     struct EnuCoor_i new_coor;
     of_calc_forwards(&new_coor, dist);
     of_move_waypoint(wp, &new_coor);
     return 0;
 }
 
 /**
  * Bereken de nieuwe positie op basis van afstand
  */
 static uint8_t of_calc_forwards(struct EnuCoor_i *coor, float dist) {
     float psi = stateGetNedToBodyEulers_f()->psi;
     coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(psi)*dist);
     coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(psi)*dist);
     return 0;
 }
 
 /**
  * Verplaats een waypoint
  */
 static uint8_t of_move_waypoint(uint8_t wp, struct EnuCoor_i *coor) {
     waypoint_move_xy_i(wp, coor->x, coor->y);
     return 0;
 }
 