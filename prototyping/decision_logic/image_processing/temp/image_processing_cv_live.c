#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <paparazzi.h>
#include <mcu_periph/uart.h>
#include <subsystems/datalink/downlink.h>
#include <subsystems/abi.h>
#include <subsystems/vision/vision.h>

#define GRID_ROWS 40
#define GRID_COLS 160

// YUV Filter Ranges for Green Detection
#define Y_MIN 90
#define Y_MAX 210
#define U_MIN 75
#define U_MAX 115
#define V_MIN 69
#define V_MAX 145

// Black & White Filtering Thresholds
#define X_WHITE 3  // Consecutive white needed to flip run to white
#define Y_BLACK 4  // Consecutive black needed to lock column as black

// Function to process each frame
typedef struct image_t image_t;

static void process_frame(image_t *img) {
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
                downsampled[j][i] = 255; // Mark as white (free space)
            } else {
                downsampled[j][i] = 0; // Mark as black (obstacle)
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
                    for (int r = run_start; r >= row; r--) downsampled[r][col] = 255;
                    run_start = -1;
                }
            }
        }
    }

    // Extract Horizon Line
    int horizon[GRID_COLS];
    for (int col = 0; col < downsample_w; col++) {
        horizon[col] = -1;
        for (int row = 0; row < downsample_h; row++) {
            if (downsampled[row][col] == 255) {
                horizon[col] = row;
                break;
            }
        }
    }

    // Send Horizon Data to Paparazzi
    DOWNLINK_SEND_HORIZON_DATA(DefaultChannel, DefaultDevice, horizon, GRID_COLS);
}

// Initialize Video Processing
void init_vision(void) {
    vision_register_process(process_frame);
}