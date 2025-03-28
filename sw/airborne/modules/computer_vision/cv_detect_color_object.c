#include "modules/computer_vision/cv_detect_color_object.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "pthread.h"

// SIMULATION
// float y_min = 78, y_max = 242;
// float u_min = 79, u_max = 125;
// float v_min = 10, v_max = 133;

// REAL ~ obtained using ~/paparazzi/prototyping/new_interactive.py
float y_min = 90, y_max = 210;
float u_min = 75, u_max = 115;
float v_min = 69, v_max = 145;

float min_black = 5; //

#define BLOCK_SIZE 5 //size of block for downsizing
#define GRID_ROWS 104 // number of rows in the downscaled image
#define GRID_COLS 48  // number of columns in the downscaled image

float oa_color_count_frac = 0.18f;

static pthread_mutex_t mutex;

struct row_white_count_t { //
    uint8_t white_counts[GRID_ROWS]; // number of white pixels in each row
    bool updated; // flag to indicate if the data has been updated
};
static struct row_white_count_t global_result;

// Main function to process the image
// This function is called in a separate thread for each image frame
// It performs the following steps:
// 1. Convert the YUV image to a binary image based on color thresholds
// 2. Downscale the binary image to reduce the size
// 3. Filter out noisy particles
// 4. Check for consecutive black pixels in each row and update the result
static void *process_image(void *arg) {
    struct image_t *img = (struct image_t *)arg;
    int width = img->w;
    int height = img->h;
    uint8_t *buf = img->buf;

    int new_width = width / BLOCK_SIZE;
    int new_height = height / BLOCK_SIZE;

    if (new_height > GRID_ROWS) new_height = GRID_ROWS; // cap to max rows and cols
    if (new_width > GRID_COLS) new_width = GRID_COLS;

    static int frame_counter = 0;
    frame_counter++; // to make terminal output more readable

    uint8_t *binary = malloc(width * height); // mask for binary image

    // Thresholding step: convert YUV to binary
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int base = y * width * 2;
            uint8_t y_val, u_val, v_val;

            if (x % 2 == 0) {
                u_val = buf[base + x * 2 + 0];
                y_val = buf[base + x * 2 + 1];
                v_val = buf[base + x * 2 + 2];
            } else {
                u_val = buf[base + x * 2 - 2];
                v_val = buf[base + x * 2 + 0];
                y_val = buf[base + x * 2 + 1];
            }

            binary[y * width + x] = (y_val >= y_min && y_val <= y_max &&
                                     u_val >= u_min && u_val <= u_max &&
                                     v_val >= v_min && v_val <= v_max) ? 1 : 0; 
        }
    }

    uint8_t *downscaled = malloc(new_width * new_height * 3);

    // Initialize downscaled image with white pixels
    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            int center_y = i * BLOCK_SIZE + BLOCK_SIZE / 2;
            int center_x = j * BLOCK_SIZE + BLOCK_SIZE / 2;
            int idx = (i * new_width + j) * 3;
    
            if (center_y < height && center_x < width && binary[center_y * width + center_x] == 1) {
                downscaled[idx] = downscaled[idx + 1] = downscaled[idx + 2] = 255; // white
            } else {
                downscaled[idx] = downscaled[idx + 1] = downscaled[idx + 2] = 0;   // black
            }
        }
    }
    
    // MIN_BLACK logic: truncate row to black if there are enough consecutive black pixels
    // this results in a matrix that does not alternate between black and white
    // this allows us to reduce the matrix to a 1D array of white pixel counts
    for (int i = 0; i < new_height; i++) {
        int consecutive_black = 0;
        for (int j = 0; j < new_width; j++) {
            int idx = (i * new_width + j) * 3;
            bool is_black = downscaled[idx] == 0 && downscaled[idx + 1] == 0 && downscaled[idx + 2] == 0;
            if (is_black) {
                consecutive_black++;
                if (consecutive_black >= min_black) {
                    for (int k = j; k < new_width; k++) {
                        int idx2 = (i * new_width + k) * 3;
                        downscaled[idx2] = downscaled[idx2 + 1] = downscaled[idx2 + 2] = 0;
                    }
                    break;
                }
            } else {
                consecutive_black = 0;
            }
        }
    }

    uint8_t white_pixel_counts[GRID_ROWS] = {0};
    for (int i = 0; i < new_height; i++) {
        int count = 0;
        for (int j = 0; j < new_width; j++) {
            int idx = (i * new_width + j) * 3;
            if (downscaled[idx] == 255 && downscaled[idx + 1] == 255 && downscaled[idx + 2] == 255) {
                count++;
            }
        }
        if (count > 254) count = 254;
        white_pixel_counts[i] = (uint8_t)count; // the 1D array of white pixel counts
    }

    // Store result in shared struct (thread-safe)
    pthread_mutex_lock(&mutex);
    memcpy(global_result.white_counts, white_pixel_counts, GRID_ROWS);
    global_result.updated = true;
    pthread_mutex_unlock(&mutex);

    // Print the downscaled matrix for debugging
    if (frame_counter % 20 == 0) {
    
        printf("Downscaled matrix:\n");
        for (int i = 0; i < new_height; i++) {
            for (int j = 0; j < new_width; j++) {
                int idx = (i * new_width + j) * 3;
                printf("%c", downscaled[idx] == 255 ? '#' : '.');
            }
            printf("\n");
        }
    }
    free(binary);
    free(downscaled);
    return NULL;
}

// Initialization function: sets up mutex and registers image callback
void color_object_detector_init(void) {
    pthread_mutex_init(&mutex, NULL);
    memset(&global_result, 0, sizeof(global_result));
    cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, process_image, 10, 0);
}

// Called periodically in main loop to use the processed image result
// Checks if middle 6 rows have enough white pixels, if not we send this to the
// orange_avoider.c
void color_object_detector_periodic(void) {
    static struct row_white_count_t local_result;
    pthread_mutex_lock(&mutex);
    memcpy(&local_result, &global_result, sizeof(struct row_white_count_t));
    pthread_mutex_unlock(&mutex);

    if (local_result.updated) {
        local_result.updated = false;

        int mid0 = GRID_ROWS / 2 - 3;
        int mid1 = GRID_ROWS / 2 - 2;
        int mid2 = GRID_ROWS / 2 - 1;
        int mid3 = GRID_ROWS / 2;
        int mid4 = GRID_ROWS / 2 + 1;
        int mid5 = GRID_ROWS / 2 + 2;

        bool condition_met = !(global_result.white_counts[mid0] > 1 &&
                               global_result.white_counts[mid1] > 1 &&
                               global_result.white_counts[mid2] > 1 &&
                               global_result.white_counts[mid3] > 1 &&
                               global_result.white_counts[mid4] > 1 &&
                               global_result.white_counts[mid5] > 1); 

        printf("[DEBUG] PERIODIC -- condition_met: %d\n", condition_met);
        AbiSendMsgHORIZON_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, condition_met);
    }
}
