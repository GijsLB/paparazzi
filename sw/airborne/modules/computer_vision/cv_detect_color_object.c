/*
 * Updated cv_detect_color_object.c to remove undefined references and ensure compilation
 */

#include "modules/computer_vision/cv_detect_color_object.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

#define GRID_ROWS 40
#define GRID_COLS 160

// YUV Filter Ranges for Ground Detection
#define Y_MIN 90
#define Y_MAX 210
#define U_MIN 75
#define U_MAX 115
#define V_MIN 69
#define V_MAX 145

// Black & White Filtering Thresholds
#define X_WHITE 3  // Consecutive white needed to flip run to white
#define Y_BLACK 4  // Consecutive black needed to lock column as black

static pthread_mutex_t mutex;

uint8_t cod_lum_min1 = 0, cod_lum_max1 = 255;
uint8_t cod_cb_min1 = 0, cod_cb_max1 = 255;
uint8_t cod_cr_min1 = 0, cod_cr_max1 = 255;

uint8_t cod_lum_min2 = 0, cod_lum_max2 = 255;
uint8_t cod_cb_min2 = 0, cod_cb_max2 = 255;
uint8_t cod_cr_min2 = 0, cod_cr_max2 = 255;

bool cod_draw1 = false;
bool cod_draw2 = false;

float oa_color_count_frac = 0.18f;

// Horizon detection data
struct horizon_t {
    uint8_t horizon[GRID_COLS];
    bool updated;
};
static struct horizon_t global_horizon;

/**
 * Process image: Downsampling, YUV filtering, and black/white logic.
 */
static void process_image(struct image_t *img) {
    uint8_t *buf = img->buf;
    int width = img->w;
    int height = img->h;
    int downsample_w = GRID_COLS;
    int downsample_h = GRID_ROWS;
    
    uint8_t downsampled[GRID_ROWS][GRID_COLS];
    memset(downsampled, 0, sizeof(downsampled));

    // Downsample the image
    for (int j = 0; j < downsample_h; j++) {
        for (int i = 0; i < downsample_w; i++) {
            int src_x = (i * width) / downsample_w;
            int src_y = (j * height) / downsample_h;
            int idx = (src_y * width + src_x) * 2; // UYVY format
            uint8_t y = buf[idx];
            uint8_t u = buf[idx + 1];
            uint8_t v = buf[idx + 3];

            // Apply YUV Filter
            if (y >= Y_MIN && y <= Y_MAX && u >= U_MIN && u <= U_MAX && v >= V_MIN && v <= V_MAX) {
                downsampled[j][i] = 1; // Mark as ground
            } else {
                downsampled[j][i] = 0; // Mark as obstacle
            }
        }
    }

    // Apply Black & White Logic
    for (int col = 0; col < downsample_w; col++) {
        int locked = 0, consecutive_black = 0, consecutive_white = 0;
        int run_start = -1;
        for (int row = downsample_h - 1; row >= 0; row--) {
            uint8_t *pixel = &downsampled[row][col];
            int is_black = (*pixel == 0);
            if (locked) {
                *pixel = 0;
                continue;
            }
            if (is_black) {
                consecutive_black++;
                consecutive_white = 0;
                if (run_start == -1) run_start = row;
                if (consecutive_black >= Y_BLACK) {
                    for (int r = run_start; r >= row; r--) downsampled[r][col] = 0;
                    locked = 1;
                }
            } else {
                consecutive_white++;
                consecutive_black = 0;
                if (consecutive_white >= X_WHITE) {
                    for (int r = run_start; r >= row; r--) downsampled[r][col] = 1;
                    run_start = -1;
                }
            }
        }
    }

    // Extract Horizon Line
    pthread_mutex_lock(&mutex);
    for (int col = 0; col < downsample_w; col++) {
        global_horizon.horizon[col] = 0xFF; // Invalid value
        for (int row = 0; row < downsample_h; row++) {
            if (downsampled[row][col] == 1) { // Ground detected
                global_horizon.horizon[col] = row;
                break;
            }
        }
    }
    global_horizon.updated = true;
    pthread_mutex_unlock(&mutex);
}

void color_object_detector_init(void) {
    pthread_mutex_init(&mutex, NULL);
    memset(&global_horizon, 0, sizeof(global_horizon));
    cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, process_image, 10, 0);
}

void color_object_detector_periodic(void) {
    static struct horizon_t local_horizon;
    pthread_mutex_lock(&mutex);
    memcpy(&local_horizon, &global_horizon, sizeof(struct horizon_t));
    pthread_mutex_unlock(&mutex);

    if (local_horizon.updated) {
        // Send ABI message
        AbiSendMsgHORIZON_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, local_horizon.horizon);
        local_horizon.updated = false;

        // Save to a log file
        FILE *logfile = fopen("/home/gijs/paparazzi/horizon_log.txt", "a"); // Open in append mode
        if (logfile) {
            for (int i = 0; i < GRID_COLS; i++) {
                fprintf(logfile, "%d ", local_horizon.horizon[i]);
            }
            fprintf(logfile, "\n"); // New line for each frame
            fclose(logfile);
        } else {
            perror("Error opening horizon_log.txt");
        }
    }
}


