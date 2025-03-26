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

// REAL
float y_min = 90, y_max = 210;
float u_min = 75, u_max = 115;
float v_min = 69, v_max = 145;

float min_black = 3;

#define BLOCK_SIZE 5
#define GRID_ROWS 104
#define GRID_COLS 48

// to do: remove all DEBUG print statements or make VERBOSE
// to do: decision logic -> if sides l&r are 0, then small nudge in other direction
// to do: if center is 0, turn to direction with most white pixels rather than random



float oa_color_count_frac = 0.18f; //weghalen

static pthread_mutex_t mutex;

struct row_white_count_t {
    uint8_t white_counts[GRID_ROWS];
    bool updated;
};
static struct row_white_count_t global_result;

static void filter_particles(uint8_t *image_data, int width, int height) {
    printf("[DEBUG] Entered filter_particles\n");
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
    printf("[DEBUG] Exiting filter_particles\n");
}

static void *process_image(void *arg) {
    printf("[DEBUG] SHIT SHIT SHIT process_image CALLED\n");
    struct image_t *img = (struct image_t *)arg;
    int width = img->w;
    int height = img->h;
    uint8_t *buf = img->buf;

    printf("[DEBUG] Image dims: %d x %d\n", width, height);

    int new_width = width / BLOCK_SIZE;
    int new_height = height / BLOCK_SIZE;

    if (new_height > GRID_ROWS) new_height = GRID_ROWS;
    if (new_width > GRID_COLS) new_width = GRID_COLS;

    static int frame_counter = 0;
    frame_counter++;

    uint8_t *binary = malloc(width * height);
    printf("[DEBUG] Allocated binary buffer at %p\n", binary);

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
    printf("[DEBUG] Allocated downscaled buffer at %p\n", downscaled);

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
                downscaled[idx] = downscaled[idx + 1] = downscaled[idx + 2] = 255;
            } else {
                downscaled[idx] = downscaled[idx + 1] = downscaled[idx + 2] = 0;
            }
        }
    }

    filter_particles(downscaled, new_width, new_height);

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
        white_pixel_counts[i] = (uint8_t)count;
    }

    pthread_mutex_lock(&mutex);
    memcpy(global_result.white_counts, white_pixel_counts, GRID_ROWS);
    global_result.updated = true;
    pthread_mutex_unlock(&mutex);

    if (frame_counter % 20 == 0) {
        printf("[DEBUG] Frame %d, YUV center sample: Y=%d U=%d V=%d\n", frame_counter, buf[width * height], buf[width * height + 1], buf[width * height + 3]);
    
        printf("[DEBUG] Downscaled matrix:\n");
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
    printf("[DEBUG] Exiting process_image\n");
    return NULL;
}

void color_object_detector_init(void) {
    pthread_mutex_init(&mutex, NULL);
    printf("[DEBUG] INIT -- Before memset\n");
    memset(&global_result, 0, sizeof(global_result));
    printf("[DEBUG] INIT -- Before cv_add_to_device\n");
    cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, process_image, 10, 0);
    printf("[DEBUG] INIT -- After cv_add_to_device\n");
}

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
