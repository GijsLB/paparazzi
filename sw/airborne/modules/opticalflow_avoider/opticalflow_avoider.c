/**
 * File: opticalflow_avoider.c
 * Optical Flow Avoider met vector callbacks voor obstakel ontwijking.
 *
 * Deze versie gaat ervan uit dat de verticale oscillatie (via altitude-aanpassing)
 * wordt geregeld door het flight plan. Deze module past alleen de horizontale (X, Y)
 * waypoint aan en controleert de flow op basis van de y-component.
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
     OF_STATE_HOVER,
     OF_STATE_FORWARD,
     OF_STATE_ROTATING
 };
 
 // ======== Parameters ========
//  static float OF_THRESHOLD = 15.f;
 float OF_THRESHOLD = 15.f;

 static float FORWARD_DIST = 1.0f;
 static float TURN_ANGLE   = 45.0f;  // 90°-rotatie als obstakel wordt gedetecteerd
 float obstacle_threshold = 0;
 #define HOVER_TIME_THRESHOLD 1.0f  // Tijd in HOVER voordat vooruit wordt gegaan
 #define FORWARD_THRESHOLD POS_BFP_OF_REAL(0.2f)  // Drempel (in fixed-point eenheden) voor het bereiken van 1 m vooruit
 
 // ======== Global vars ========
 static enum of_state_t of_state = OF_STATE_HOVER;
 static int32_t motion_x = 0;
 static int32_t motion_y = 0;
 static double motion_magnitude = 0.0;
 static float last_state_start = 0.f;
 static int rotation_started = 0;  // Zorgt dat de 90° rotatie slechts één keer gestart wordt
 
 // ======== Forward-declaraties ========
 static void of_set_state(enum of_state_t new_state);
 static uint8_t of_increase_heading(float delta_deg);
 static uint8_t of_move_waypoint_forward(uint8_t wp, float dist);
 static uint8_t of_calc_forwards(struct EnuCoor_i *coor, float dist);
 static uint8_t of_move_waypoint(uint8_t wp, struct EnuCoor_i *coor);
 
 static void of_vector_callback(uint8_t sender_id,
                 uint8_t count,
                 int32_t *flow_xy);
 
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
     of_set_state(OF_STATE_HOVER);
 }
 
 /**
  * opticalflow_avoider_periodic
  *
  * Gedrag:
  * - In OF_STATE_HOVER wordt horizontaal enkel gecontroleerd of er een obstakel is.
  *   Als er 3 seconden geen obstakel is, schakelt hij over naar FORWARD.
  * - In OF_STATE_FORWARD wordt de horizontale voorwaartse beweging (1 m vooruit) éénmalig uitgevoerd.
  *   Pas als de drone de doelpositie heeft bereikt, keert hij terug naar HOVER.
  * - In OF_STATE_ROTATING wordt een 90°-rotatie gestart.
  *   Zodra de rotatie afgerond is en er minstens 2 seconden gewacht is, keert hij terug naar HOVER.
  */
 void opticalflow_avoider_periodic(void) {
     if (!autopilot_in_flight() || autopilot_get_mode() != AP_MODE_NAV) {
         return;
     }
 
     float now_s = get_sys_time_float();
     float dt_s  = now_s - last_state_start;
 
     switch (of_state) {
         case OF_STATE_HOVER: {
             // In HOVER controleren we alleen of er een obstakel is.
             if (fabsf(motion_magnitude) > OF_THRESHOLD) {
                 OF_PRINT("Obstacle gedetecteerd: magnitude=%.2f, overschakelen naar ROTATING\n", motion_magnitude);
                 of_set_state(OF_STATE_ROTATING);
             } else if (dt_s > HOVER_TIME_THRESHOLD) {
                 OF_PRINT("Geen obstakel gedurende %.2f sec, overschakelen naar FORWARD\n", dt_s);
                 of_set_state(OF_STATE_FORWARD);
             }
             break;
         }
         case OF_STATE_FORWARD: {
             // Forward state: horizontaal 1 meter vooruit.
             static struct EnuCoor_i target_coor;
             static int forward_initialized = 0;
             if (!forward_initialized) {
                 of_calc_forwards(&target_coor, FORWARD_DIST);
                 of_move_waypoint_forward(WP_GOAL, FORWARD_DIST);
                 forward_initialized = 1;
                 OF_PRINT("Forward: target waypoint ingesteld: x=%d, y=%d\n", target_coor.x, target_coor.y);
             }
             // Vergelijk huidige horizontale positie met target.
             struct EnuCoor_i *cur = stateGetPositionEnu_i();
             int32_t dx = cur->x - target_coor.x;
             int32_t dy = cur->y - target_coor.y;
             int32_t dist_fp = (int32_t) sqrt((double)(dx * dx + dy * dy));
             if (dist_fp < FORWARD_THRESHOLD) {
                 OF_PRINT("Forward waypoint bereikt: afstand=%d (threshold=%d)\n", dist_fp, FORWARD_THRESHOLD);
                 forward_initialized = 0;
                 of_set_state(OF_STATE_HOVER);
             }
             break;
         }
         case OF_STATE_ROTATING: {
             if (!rotation_started) {
                 OF_PRINT("Start 45° rotatie\n");
                 of_increase_heading(TURN_ANGLE);
                 rotation_started = 1;
             }
             float current_psi = stateGetNedToBodyEulers_f()->psi;
             float diff = current_psi - nav.heading;
             FLOAT_ANGLE_NORMALIZE(diff);
             if (fabsf(diff) < RadOfDeg(5.f)) {
                 // Wacht nu 2 seconden nadat de rotatie afgerond is.
                 if (dt_s >= 2.0f) {
                     OF_PRINT("Rotatie afgerond en 2 sec gewacht, terug naar HOVER\n");
                     of_set_state(OF_STATE_HOVER);
                 } else {
                     OF_PRINT("Rotatie afgerond, wacht nog %.2f sec\n", 2.0f - dt_s);
                 }
             }
             if (dt_s > 8.0f) {
                 OF_PRINT("WARNING: Rotatie te lang, terug naar HOVER\n");
                 of_set_state(OF_STATE_HOVER);
             }
             break;
         }
     }
 }
 
// Vergelijkingsfunctie voor qsort (aflopend)
static int cmp_desc(const void* a, const void* b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return 1;
    if (da > db) return -1;
    return 0;
}

static void of_vector_callback(uint8_t sender_id,
    uint8_t count,
    int32_t *flow_xy)
{
    fprintf(stderr, "[OF] cb: got count=%d from sender_id=%d\n", count, sender_id);

    int64_t sum_fx = 0;
    // Maak een array voor de absolute y-waarden van de vectoren waarvan y niet nul is.
    double valid_abs_y[count];  // Maximale grootte is count
    int valid_count = 0;
    for (int i = 0; i < count; i++) {
        int32_t fx = flow_xy[2 * i];
        int32_t fy = flow_xy[2 * i + 1];
        sum_fx += fx;
        // Alleen toevoegen als fy != 0
        if (fy != 0) {
            valid_abs_y[valid_count++] = fabs((double)fy);
        }
        fprintf(stderr, "   i=%d => flow_x=%ld flow_y=%ld\n", i, (long)fx, (long)fy);
    }

    if (count > 0) {
        motion_x = sum_fx / count;
        if (valid_count > 0) {
            // Sorteer de geldige absolute y-waarden in aflopende volgorde.
            qsort(valid_abs_y, valid_count, sizeof(double), cmp_desc);
            // Neem de top 5 waarden (of minder als er minder dan 5 beschikbaar zijn)
            int n = valid_count >= 5 ? 5 : valid_count;
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
        fprintf(stderr, "[OF] Flow magnitude too high (%.2f), discarding measurement\n", motion_magnitude);
        motion_x = 0;
        motion_magnitude = 100;
    }

    fprintf(stderr, "[OF] average fx = %ld, average magnitude (top 5 nonzero |fy|) = %.2f\n",
            (long)motion_x, motion_magnitude);
}




 
 /**
  * Wijzigt de state en reset de timer.
  */
 static void of_set_state(enum of_state_t new_state) {
     of_state = new_state;
     last_state_start = get_sys_time_float();
     if (new_state == OF_STATE_HOVER) {
         rotation_started = 0;  // Reset de rotatieflag bij terugkeer naar HOVER
     }
     OF_PRINT("Switch state => %d\n", (int)new_state);
 }
 
 /**
  * Verhoogt de heading met delta_deg.
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
  * Verplaatst het waypoint 1 meter vooruit (alleen horizontaal).
  */
 static uint8_t of_move_waypoint_forward(uint8_t wp, float dist) {
     struct EnuCoor_i new_coor;
     of_calc_forwards(&new_coor, dist);
     of_move_waypoint(wp, &new_coor);
     return 0;
 }
 
 /**
  * Berekent de nieuwe positie (x, y) op basis van afstand.
  */
 static uint8_t of_calc_forwards(struct EnuCoor_i *coor, float dist) {
     float psi = stateGetNedToBodyEulers_f()->psi;
     coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(psi) * dist);
     coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(psi) * dist);
     return 0;
 }
 
 /**
  * Verplaatst een waypoint (x, y) naar de berekende positie.
  */
 static uint8_t of_move_waypoint(uint8_t wp, struct EnuCoor_i *coor) {
     waypoint_move_xy_i(wp, coor->x, coor->y);
     return 0;
 }
 