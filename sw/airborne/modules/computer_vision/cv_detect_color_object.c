#include "modules/computer_vision/cv_detect_color_object.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "pthread.h"

#define Y_MIN 78
#define Y_MAX 242
#define U_MIN 79
#define U_MAX 150
#define V_MIN 50
#define V_MAX 133

#define BLOCK_SIZE 5
#define GRID_ROWS 333
#define GRID_COLS 333

// Dummy vars (needed for compilation, not used)
uint8_t cod_lum_min1 = 0, cod_lum_max1 = 255;
uint8_t cod_cb_min1 = 0, cod_cb_max1 = 255;
uint8_t cod_cr_min1 = 0, cod_cr_max1 = 255;
uint8_t cod_lum_min2 = 0, cod_lum_max2 = 255;
uint8_t cod_cb_min2 = 0, cod_cb_max2 = 255;
uint8_t cod_cr_min2 = 0, cod_cr_max2 = 255;
bool cod_draw1 = false;
bool cod_draw2 = false;
float oa_color_count_frac = 0.18f;

static pthread_mutex_t mutex;

struct row_white_count_t {
    uint8_t white_counts[GRID_ROWS];
    bool updated;
};
static struct row_white_count_t global_result;

static void filter_particles(uint8_t *image_data, int width, int height) {
    uint8_t *filtered_image = malloc(width * height * 3);
    memcpy(filtered_image, image_data, width * height * 3);

    for (int i = 0; i < height; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = (i * width + j) * 3;

            int current_pixel = (image_data[idx] < 128) ? 0 : 1;
            int opposite_pixel = 1 - current_pixel;

            int left_pixel = (image_data[((i * width + j - 1) * 3)] < 128) ? 0 : 1;
            int right_pixel = (image_data[((i * width + j + 1) * 3)] < 128) ? 0 : 1;

            int opposite_count = 0;
            if (left_pixel == opposite_pixel) opposite_count++;
            if (right_pixel == opposite_pixel) opposite_count++;

            if (opposite_count == 2) {
                filtered_image[idx]     = (opposite_pixel == 0) ? 0 : 255;
                filtered_image[idx + 1] = (opposite_pixel == 0) ? 0 : 255;
                filtered_image[idx + 2] = (opposite_pixel == 0) ? 0 : 255;
            }
        }
    }

    memcpy(image_data, filtered_image, width * height * 3);
    free(filtered_image);
}

static void process_image(struct image_t *img) {
    int width = img->w;
    int height = img->h;
    uint8_t *buf = img->buf;

    int new_width = width / BLOCK_SIZE;
    int new_height = height / BLOCK_SIZE;

    if (new_height > GRID_ROWS) new_height = GRID_ROWS;
    if (new_width > GRID_COLS) new_width = GRID_COLS;

    static int frame_counter = 0;
    frame_counter++;

    uint8_t *binary = malloc(width * height);
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

            binary[y * width + x] = (y_val >= Y_MIN && y_val <= Y_MAX &&
                                     u_val >= U_MIN && u_val <= U_MAX &&
                                     v_val >= V_MIN && v_val <= V_MAX) ? 1 : 0;
        }
    }

    uint8_t *downscaled = malloc(new_width * new_height * 3);
    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            int white_count = 0;
            for (int yb = i * BLOCK_SIZE; yb < (i + 1) * BLOCK_SIZE; yb++) {
                for (int xb = j * BLOCK_SIZE; xb < (j + 1) * BLOCK_SIZE; xb++) {
                    if (binary[yb * width + xb] == 1) {
                        white_count++;
                    }
                }
            }
            int idx = (i * new_width + j) * 3;
            if (white_count > 12) {
                downscaled[idx]     = 255;
                downscaled[idx + 1] = 255;
                downscaled[idx + 2] = 255;
            } else {
                downscaled[idx]     = 0;
                downscaled[idx + 1] = 0;
                downscaled[idx + 2] = 0;
            }
        }
    }

    filter_particles(downscaled, new_width, new_height);

    for (int i = 0; i < new_height; i++) {
        int found_black = 0;
        for (int j = 0; j < new_width; j++) {
            int idx = (i * new_width + j) * 3;
            if (downscaled[idx] == 0 && downscaled[idx + 1] == 0 && downscaled[idx + 2] == 0) {
                found_black = 1;
            }
            if (found_black) {
                downscaled[idx]     = 0;
                downscaled[idx + 1] = 0;
                downscaled[idx + 2] = 0;
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
        white_pixel_counts[i] = (uint8_t)count;
    }

    pthread_mutex_lock(&mutex);
    memcpy(global_result.white_counts, white_pixel_counts, GRID_ROWS);
    global_result.updated = true;
    pthread_mutex_unlock(&mutex);

    if (frame_counter % 20 == 0) {
        int idx = (height / 2 * width + width / 2) * 2;
        printf("Sample YUV at center: Y=%d, U=%d, V=%d\n", buf[idx], buf[idx + 1], buf[idx + 3]);

        printf("White pixel counts per row:\n[");
        for (int i = 0; i < new_height; i++) {
            printf("%d", white_pixel_counts[i]);
            if (i < new_height - 1) printf(", ");
        }
        printf("]\n");

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
}

void color_object_detector_init(void) {
    pthread_mutex_init(&mutex, NULL);
    memset(&global_result, 0, sizeof(global_result));
    cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, process_image, 10, 0);
}
 
void color_object_detector_periodic(void) {
    static struct row_white_count_t local_result;
    pthread_mutex_lock(&mutex);
    memcpy(&local_result, &global_result, sizeof(struct row_white_count_t));
    pthread_mutex_unlock(&mutex);

    if (local_result.updated) {
        local_result.updated = false;

        // Check if the four middle values are all greater than 1
        int mid1 = GRID_ROWS / 2 - 2;
        int mid2 = GRID_ROWS / 2 - 1;
        int mid3 = GRID_ROWS / 2;
        int mid4 = GRID_ROWS / 2 + 1;

        bool condition_met = (local_result.white_counts[mid1] > 1 &&
                               local_result.white_counts[mid2] > 1 &&
                               local_result.white_counts[mid3] > 1 &&
                               local_result.white_counts[mid4] > 1);

        // Send ABI message with True/False
        AbiSendMsgHORIZON_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, condition_met);
    }
}