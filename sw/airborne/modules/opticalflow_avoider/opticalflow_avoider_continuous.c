/**
 * File: opticalflow_avoider.c
 * Purpose: Robust obstacle avoidance using optical flow for autonomous drone navigation
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
 #include "math/pprz_algebra_int.h"
 
 // ======== Debugging macro ========
 #define OF_VERBOSE TRUE
 #if OF_VERBOSE
 #define OF_PRINT(fmt, ...) fprintf(stderr, "[OF Avoider->%s()] " fmt, __FUNCTION__, ##__VA_ARGS__)
//  #else
//  #define OF_PRINT(...) {}
 #endif
 
 // ======== Critical Parameters ========
 #define MAX_INDIVIDUAL_FLOW_MAG 30.0f    // Maximum allowed magnitude for individual flow vectors
 #define MIN_VALID_VECTORS 5              // Minimum number of valid vectors needed for reliable detection
 #define MOVING_AVERAGE_SIZE 3            // Window size for temporal smoothing
//  #define DANGER_THRESHOLD 45.0f           // Higher threshold for immediate action
//  #define CAUTION_THRESHOLD 25.0f           // Lower threshold for gradual response
 #define BASE_FORWARD_SPEED 1.0f          // Increased base forward speed
 #define MIN_FORWARD_SPEED 0.6f           // Minimum speed when near obstacles
 #define MIN_TURN_ANGLE 45.0f             // Minimum turn to avoid obstacle
 #define MAX_TURN_ANGLE 120.0f            // Maximum turn when danger is high
 #define SECTOR_SAFETY_MARGIN 0.0f       // Safety margin from sector boundaries (meters)
 #define OSCILLATION_PERIOD 1.5f          // Faster oscillation for quicker flow detection
 #define OSCILLATION_AMPLITUDE 10.0f      // Reduced oscillation to maintain forward motion
 #define PROGRESS_CHECK_INTERVAL 1.0f     // Check progress more frequently
 #define MIN_PROGRESS_DIST 0.3f           // Reduced minimum progress threshold
 #define TURN_BOOST_FACTOR 1.5f           // Boost turn speed when needed
 #define FLOW_CONFIDENCE_THRESHOLD 0.7f   // Minimum confidence to react to flow
 #define MIN_FLOW_THRESHOLD 2.0f          // Minimum flow magnitude to consider valid
 #define POST_TURN_CHECK_TIME 0.2f        // Time to wait after turn before checking
 #define MAX_CIRCLE_TIME 4.0f             // Maximum time to allow for rotation

 #ifndef CAUTION_THRESHOLD
 #define CAUTION_THRESHOLD 35.0f
 #endif
 float caution_threshold = CAUTION_THRESHOLD;
 
 #ifndef DANGER_THRESHOLD
 #define DANGER_THRESHOLD 90.0f
 #endif
 float danger_threshold = DANGER_THRESHOLD;

 // ======== State Machine ========
 enum of_state_t {
     OF_STATE_WAIT_FORWARD,
     OF_STATE_CHECK_OBSTACLE,
     OF_STATE_ROTATING,
     OF_STATE_OSCILLATING,
     OF_STATE_BOUNDARY_TURN  // <-- new state
 };
 
 // ======== Global State Variables ========
 float obstacle_threshold;
 static enum of_state_t of_state;
 static int32_t motion_x;
 static int32_t motion_y;
 static double motion_magnitude;
 static float last_state_start;
 static float oscillation_start_time;
 static float base_heading;
 static float current_forward_speed;
 static uint32_t distance_traveled;
 static float last_check_time;
 static struct FloatVect2 last_pos;
 static float confidence_level;
 static float last_pos_x;
 static float last_pos_y;
 static float last_progress_check;
 static int consecutive_turns;
 static int last_turn_direction;
 
 // Motion smoothing variables
 static double motion_history[MOVING_AVERAGE_SIZE];
 static int history_index;
 static double smoothed_magnitude;
 
 // Boundary variables (from red sector)
 static float cyber_min_x, cyber_max_x;
 static float cyber_min_y, cyber_max_y;
 
 // ABI event
 static abi_event of_vec_ev;
 
 // Function declarations
 static void of_set_state(enum of_state_t new_state);
 static uint8_t of_increase_heading(float delta_deg);
 static uint8_t of_move_waypoint_forward(uint8_t wp, float dist);
 static uint8_t of_calc_forwards(struct EnuCoor_i *coor, float dist);
 static uint8_t of_move_waypoint(uint8_t wp, struct EnuCoor_i *coor);
 static void update_motion_average(double new_magnitude);
 static float get_turn_angle(float magnitude);
 static void of_vector_callback(uint8_t sender_id, uint8_t count, int32_t *flow_xy);
 static bool check_sector_boundaries(void);
 static float get_oscillation_heading(float base_heading, float elapsed_time);
 
 /* --- Helper: Compare two doubles in descending order --- */
 static int cmp_desc(const void* a, const void* b) {
     double da = *(const double *)a;
     double db = *(const double *)b;
     return (da < db) ? 1 : (da > db ? -1 : 0);
 }
 
 /**
  * Set new state and update timestamp
  */
 static void of_set_state(enum of_state_t new_state) {
     of_state = new_state;
     last_state_start = get_sys_time_float();
     // Reset rotation flag when returning to WAIT_FORWARD
     if (new_state == OF_STATE_WAIT_FORWARD)
         last_turn_direction = 1;
     OF_PRINT("Switch state => %d\n", (int)new_state);
 }
 
 /**
  * Check if we're approaching sector boundaries and need to turn.
  */
 static bool check_sector_boundaries(void) {
     struct EnuCoor_f *pos = stateGetPositionEnu_f();
     if (pos->x < cyber_min_x + SECTOR_SAFETY_MARGIN ||
         pos->x > cyber_max_x - SECTOR_SAFETY_MARGIN ||
         pos->y < cyber_min_y + SECTOR_SAFETY_MARGIN ||
         pos->y > cyber_max_y - SECTOR_SAFETY_MARGIN) {
         return true;
     }
     return false;
 }
 
 /**
  * Calculate oscillating heading based on time.
  */
 static float get_oscillation_heading(float base_heading, float elapsed_time) {
     float phase = (2.0f * M_PI * elapsed_time) / OSCILLATION_PERIOD;
     float delta_heading = OSCILLATION_AMPLITUDE * sinf(phase);
     return base_heading + RadOfDeg(delta_heading);
 }
 
 /**
  * Update moving average of flow measurements.
  */
 static void update_motion_average(double new_magnitude) {
     motion_history[history_index] = new_magnitude;
     history_index = (history_index + 1) % MOVING_AVERAGE_SIZE;
     smoothed_magnitude = 0;
     for (int i = 0; i < MOVING_AVERAGE_SIZE; i++) {
         smoothed_magnitude += motion_history[i];
     }
     smoothed_magnitude /= MOVING_AVERAGE_SIZE;
 }
 
 /**
  * Determines turn direction and magnitude based on flow and boundaries.
  */
 static float get_turn_angle(float magnitude) {
     struct EnuCoor_f *pos = stateGetPositionEnu_f();
     float turn_angle;
     if (pos->x < SECTOR_SAFETY_MARGIN)
         last_turn_direction = 1;
     else if (pos->x > (5.0f - SECTOR_SAFETY_MARGIN))
         last_turn_direction = -1;
     else
         last_turn_direction *= -1;
     turn_angle = MIN_TURN_ANGLE + (magnitude - caution_threshold) *
                  (MAX_TURN_ANGLE - MIN_TURN_ANGLE) / (danger_threshold - caution_threshold);
     turn_angle = fminf(fmaxf(turn_angle, MIN_TURN_ANGLE), MAX_TURN_ANGLE);
     return turn_angle * last_turn_direction;
 }
 
 /**
  * Check if we're making forward progress.
  */
 static bool check_forward_progress(void) {
     struct EnuCoor_f *pos = stateGetPositionEnu_f();
     float dx = pos->x - last_pos_x;
     float dy = pos->y - last_pos_y;
     float dist = sqrtf(dx * dx + dy * dy);
     float now = get_sys_time_float();
     if (now - last_progress_check > 2.0f) {
         last_progress_check = now;
         last_pos_x = pos->x;
         last_pos_y = pos->y;
         if (dist < MIN_PROGRESS_DIST) {
             consecutive_turns++;
             return false;
         } else {
             consecutive_turns = 0;
             return true;
         }
     }
     return true;
 }
 
 /**
  * Calculate current confidence level in flow measurements.
  */
 static float calculate_confidence(int valid_count, int total_count, float max_flow) {
     float count_ratio = (float)valid_count / total_count;
     float flow_quality = (max_flow < MAX_INDIVIDUAL_FLOW_MAG) ? 1.0f : 0.5f;
     return count_ratio * flow_quality;
 }
 
 /**
  * Adjust forward speed based on obstacle proximity and confidence.
  */
 static float get_adjusted_forward_speed(float flow_magnitude) {
     float speed_factor = 1.0f - (flow_magnitude / danger_threshold);
     speed_factor = fminf(fmaxf(speed_factor, 0.0f), 1.0f);
     float adjusted_speed = MIN_FORWARD_SPEED + (BASE_FORWARD_SPEED - MIN_FORWARD_SPEED) * speed_factor;
     if (!check_forward_progress())
         adjusted_speed *= TURN_BOOST_FACTOR;
     return adjusted_speed;
 }
 
 /**
  * Update distance traveled.
  */
 static void update_distance_traveled(void) {
     struct EnuCoor_f *pos = stateGetPositionEnu_f();
     float dx = pos->x - last_pos.x;
     float dy = pos->y - last_pos.y;
     float dist = sqrtf(dx * dx + dy * dy);
     float now = get_sys_time_float();
     if (now - last_check_time > PROGRESS_CHECK_INTERVAL) {
         distance_traveled += (uint32_t)(dist * 100); // centimeters
         last_pos.x = pos->x;
         last_pos.y = pos->y;
         last_check_time = now;
         OF_PRINT("Distance traveled: %.2fm\n", distance_traveled / 100.0f);
     }
 }
 
 /**
  * Optical flow vector callback:
  * Only uses the y component: it stores the absolute values of fy (if nonzero),
  * then sorts them in descending order and averages the top five.
  */

static void of_vector_callback(uint8_t sender_id, uint8_t count, int32_t *flow_xy) {
    // --- Compute raw flow magnitudes statistics ---
    double raw_min = 1e6, raw_max = 0;
    double sum_magnitudes = 0;
    for (int i = 0; i < count; i++) {
        int32_t fx = flow_xy[2 * i];
        int32_t fy = flow_xy[2 * i + 1];
        double magnitude = sqrt((double)fx * fx + (double)fy * fy);
        if (magnitude < raw_min) raw_min = magnitude;
        if (magnitude > raw_max) raw_max = magnitude;
        sum_magnitudes += magnitude;
        // Printing each vector's raw magnitude
        // OF_PRINT("  i=%d: fx=%ld, fy=%ld, raw magnitude=%.2f\n", i, (long)fx, (long)fy, magnitude);
    }
    double raw_avg = sum_magnitudes / count;
    // OF_PRINT("Raw flow stats: min=%.2f, max=%.2f, avg=%.2f\n", raw_min, raw_max, raw_avg);

    int64_t sum_fx = 0;
    double valid_abs_y[count];
    int valid_count = 0;

    for (int i = 0; i < count; i++) {
        int32_t fx = flow_xy[2 * i];
        int32_t fy = flow_xy[2 * i + 1];
        sum_fx += fx;
        if (fy != 0) {
            valid_abs_y[valid_count++] = fabs((double)fy);
        }
        // Printing each vector's fx and fy
        // OF_PRINT("  i=%d: fx=%ld, fy=%ld\n", i, (long)fx, (long)fy);
    }

    if (count > 0) {
        motion_x = sum_fx / count;
        if (valid_count > 0) {
            qsort(valid_abs_y, valid_count, sizeof(double), cmp_desc);
            int n = (valid_count >= 5) ? 5 : valid_count;
            double sum_top = 0.0;
            for (int i = 0; i < n; i++) {
                sum_top += valid_abs_y[i];
            }
            motion_magnitude = sum_top / n;
        } else {
            motion_magnitude = 0;
        }
    } else {
        motion_x = 0;
        motion_magnitude = 0;
    }

    if (motion_magnitude > 100.0) {
        OF_PRINT("Flow magnitude too high (%.2f), discarding measurement\n", motion_magnitude);
        motion_x = 0;
        motion_magnitude = 100;
    }

    // Final summary print only:
    OF_PRINT("[%f] Average fx=%ld, averaged top 5 |fy|=%.2f\n",
             get_sys_time_float(), (long)motion_x, motion_magnitude);
}

 
 /**
  * Increase heading by delta degrees.
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
  * Move waypoint forward by given distance.
  * If the computed waypoint lies outside the boundary, return a special code (1)
  * so the main state machine can execute a boundary turn.
  */
 static uint8_t of_move_waypoint_forward(uint8_t wp, float dist) {
     // 1) Compute the desired new waypoint (in fixed-point)
     struct EnuCoor_i new_coor;
     of_calc_forwards(&new_coor, dist);
 
     // 2) Convert to float for boundary checking
     float new_x = POS_FLOAT_OF_BFP(new_coor.x);
     float new_y = POS_FLOAT_OF_BFP(new_coor.y);
 
     // 3) Check if it lies outside the boundary
     bool clamped = false;
     if (new_x < cyber_min_x) {
         new_x = cyber_min_x;
         clamped = true;
     } else if (new_x > cyber_max_x) {
         new_x = cyber_max_x;
         clamped = true;
     }
     if (new_y < cyber_min_y) {
         new_y = cyber_min_y;
         clamped = true;
     } else if (new_y > cyber_max_y) {
         new_y = cyber_max_y;
         clamped = true;
     }
 
     // If a boundary clamp is detected, return a special code
     if (clamped) {
         OF_PRINT("Boundary clamp detected => returning 1\n");
         return 1;
     }
 
     // 4) If not clamped, perform normal forward movement
     new_coor.x = POS_BFP_OF_REAL(new_x);
     new_coor.y = POS_BFP_OF_REAL(new_y);
     waypoint_move_xy_i(wp, new_coor.x, new_coor.y);
     return 0; // Normal success
 }
 
 /**
  * Calculate new position based on distance.
  */
 static uint8_t of_calc_forwards(struct EnuCoor_i *coor, float dist) {
     float psi = stateGetNedToBodyEulers_f()->psi;
     coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(psi) * dist);
     coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(psi) * dist);
     return 0;
 }
 
 /**
  * Move a waypoint.
  */
 static uint8_t of_move_waypoint(uint8_t wp, struct EnuCoor_i *coor) {
     waypoint_move_xy_i(wp, coor->x, coor->y);
     return 0;
 }
 
 /**
  * Main periodic function for obstacle avoidance.
  */
 void opticalflow_avoider_periodic(void) {
     if (!autopilot_in_flight() || autopilot_get_mode() != AP_MODE_NAV) {
         return;
     }
 
     float now_s = get_sys_time_float();
     float dt_s = now_s - last_state_start;
     bool near_boundary = check_sector_boundaries();
     update_distance_traveled();
     update_motion_average(motion_magnitude);

    // --- Log overall state info ---
    OF_PRINT("Periodic: dt=%.2f, of_state=%d, current forward speed=%.2f, smoothed magnitude=%.2f\n",
        dt_s, (int)of_state, current_forward_speed, (float)smoothed_magnitude);
 
     bool reliable_flow = (confidence_level > FLOW_CONFIDENCE_THRESHOLD);
     bool obstacle_detected = reliable_flow && (smoothed_magnitude > caution_threshold);
 
     switch (of_state) {
         case OF_STATE_WAIT_FORWARD: {
             OF_PRINT("[OF_STATE_WAIT_FORWARD] dt=%.2f, obstacle=%d, boundary=%d\n",
                      dt_s, obstacle_detected, near_boundary);
             int ret = of_move_waypoint_forward(WP_GOAL, current_forward_speed);
             if (ret == 1) {
                 // Boundary clamp detected => switch to boundary-turn state
                 of_set_state(OF_STATE_BOUNDARY_TURN);
                 break;
             }
             if (dt_s > POST_TURN_CHECK_TIME) {
                 if (obstacle_detected || near_boundary) {
                     of_set_state(OF_STATE_CHECK_OBSTACLE);
                 } else {
                     current_forward_speed = get_adjusted_forward_speed(smoothed_magnitude);
                     of_move_waypoint_forward(WP_GOAL, current_forward_speed);
                     if (!near_boundary && reliable_flow) {
                         of_set_state(OF_STATE_OSCILLATING);
                         oscillation_start_time = now_s;
                         base_heading = stateGetNedToBodyEulers_f()->psi;
                     }
                 }
             }
             break;
         }
 
         case OF_STATE_OSCILLATING: {
             OF_PRINT("[OF_STATE_OSCILLATING] dt=%.2f, obstacle=%d, boundary=%d\n",
                      dt_s, obstacle_detected, near_boundary);
             if (obstacle_detected || near_boundary) {
                 of_set_state(OF_STATE_CHECK_OBSTACLE);
             } else {
                 float new_heading = get_oscillation_heading(base_heading, now_s - oscillation_start_time);
                 nav.heading = new_heading;
                 current_forward_speed = get_adjusted_forward_speed(0.1 * smoothed_magnitude);
                 of_move_waypoint_forward(WP_GOAL, current_forward_speed);
             }
             break;
         }
 
         case OF_STATE_CHECK_OBSTACLE: {
             OF_PRINT("[OF_STATE_CHECK_OBSTACLE] dt=%.2f, obstacle=%d, boundary=%d\n",
                      dt_s, obstacle_detected, near_boundary);
             int ret = of_move_waypoint_forward(WP_GOAL, current_forward_speed);
             if (ret == 1) {
                 // Boundary clamp detected => switch to boundary-turn state
                 of_set_state(OF_STATE_BOUNDARY_TURN);
                 break;
             }
             if (obstacle_detected || near_boundary) {
                 float turn_angle = get_turn_angle(smoothed_magnitude);
                 of_increase_heading(turn_angle);
                 of_set_state(OF_STATE_ROTATING);
             } else {
                 current_forward_speed = get_adjusted_forward_speed(smoothed_magnitude);
                 of_move_waypoint_forward(WP_GOAL, current_forward_speed);
                 of_set_state(OF_STATE_WAIT_FORWARD);
             }
             break;
         }

         case OF_STATE_ROTATING: {
            OF_PRINT("[OF_STATE_ROTATING] dt=%.2f\n", dt_s);
            float current_psi = stateGetNedToBodyEulers_f()->psi;
            float diff = current_psi - nav.heading;
            FLOAT_ANGLE_NORMALIZE(diff);

            // --- Log current heading information ---
            OF_PRINT("ROTATING state: current_psi=%.2f deg, nav.heading=%.2f deg, heading error=%.2f deg\n",
                DegOfRad(current_psi), DegOfRad(nav.heading), DegOfRad(diff));

            // During rotation, ignore the optical flow magnitude by using 0 for speed adjustment.
            // This prevents massive optical flow values (from turning) from affecting obstacle detection.
            if (fabsf(diff) < RadOfDeg(5.f)) {
                 // Set forward speed based on a 0 flow magnitude (i.e. no obstacle-induced slowdown)
                 current_forward_speed = get_adjusted_forward_speed(0);
                 of_move_waypoint_forward(WP_GOAL, current_forward_speed);
                 of_set_state(OF_STATE_WAIT_FORWARD);
            }
            if (dt_s > MAX_CIRCLE_TIME) {
                 current_forward_speed = MIN_FORWARD_SPEED;
                 of_move_waypoint_forward(WP_GOAL, current_forward_speed);
                 of_set_state(OF_STATE_WAIT_FORWARD);
            }
            break;
        }
        
 
         case OF_STATE_BOUNDARY_TURN: {
             OF_PRINT("[OF_STATE_BOUNDARY_TURN] Doing stepwise boundary turn.\n");
 
             // 1) Stop forward motion: set the goal waypoint to the current position.
             NavSetWaypointHere(WP_GOAL);
 
             // 2) Turn 180 degrees.
             of_increase_heading(120.0f);
 
             // 3) Move forward 0.5 m in the new direction.
             of_move_waypoint_forward(WP_GOAL, 0.6f);
 
             // 4) Switch back to WAIT_FORWARD to resume normal behavior.
             of_set_state(OF_STATE_WAIT_FORWARD);
             break;
         }
     } // End of switch
 }
 
 /**
  * Initialize the optical flow avoider.
  */
 void opticalflow_avoider_init(void) {
     OF_PRINT("Initializing optical flow avoider\n");
 
     // Initialize state variables
     of_state = OF_STATE_WAIT_FORWARD;
     motion_x = 0;
     motion_y = 0;
     motion_magnitude = 0;
     last_state_start = 0;
     history_index = 0;
     smoothed_magnitude = 0;
     last_turn_direction = 1;
     current_forward_speed = BASE_FORWARD_SPEED;
     distance_traveled = 0;
     confidence_level = 10.0f;
     consecutive_turns = 0;
     last_progress_check = 0;
 
     // Initialize position tracking
     struct EnuCoor_f *pos = stateGetPositionEnu_f();
     if (pos != NULL) {
         last_pos.x = pos->x;
         last_pos.y = pos->y;
         last_check_time = get_sys_time_float();
         last_pos_x = pos->x;
         last_pos_y = pos->y;
     } else {
         OF_PRINT("Error: stateGetPositionEnu_f() returned NULL\n");
         last_pos.x = 0;
         last_pos.y = 0;
         last_check_time = get_sys_time_float();
         last_pos_x = 0;
         last_pos_y = 0;
     }
 
     // 1) Fetch local coordinates of the 4 corners from the flight plan
     float x1 = WaypointX(WP__OZ1);
     float y1 = WaypointY(WP__OZ1);
 
     float x2 = WaypointX(WP__OZ2);
     float y2 = WaypointY(WP__OZ2);
 
     float x3 = WaypointX(WP__OZ3);
     float y3 = WaypointY(WP__OZ3);
 
     float x4 = WaypointX(WP__OZ4);
     float y4 = WaypointY(WP__OZ4);
 
     // 2) Compute min and max boundaries
     cyber_min_x = fminf(x1, fminf(x2, fminf(x3, x4)));
     cyber_max_x = fmaxf(x1, fmaxf(x2, fmaxf(x3, x4)));
     cyber_min_y = fminf(y1, fminf(y2, fminf(y3, y4)));
     cyber_max_y = fmaxf(y1, fmaxf(y2, fmaxf(y3, y4)));
 
     OF_PRINT("CyberZoo bounding box: X in [%.2f, %.2f], Y in [%.2f, %.2f]\n",
              cyber_min_x, cyber_max_x, cyber_min_y, cyber_max_y);
 
     OF_PRINT("OZ1 local coords: x=%.2f y=%.2f\n", x1, y1);
     OF_PRINT("OZ2 local coords: x=%.2f y=%.2f\n", x2, y2);
 
     // Clear motion history
     for (int i = 0; i < MOVING_AVERAGE_SIZE; i++) {
         motion_history[i] = 0;
     }
 
     // Bind to optical flow messages
     AbiBindMsgOPTICAL_FLOW_VECTORS(OPTICAL_FLOW_CALCULATOR_ID,
                                    &of_vec_ev,
                                    of_vector_callback);
 
     OF_PRINT("OpticalFlow ABI bound, waiting for messages...\n");
 }
 
 /* --- End of file --- */
 