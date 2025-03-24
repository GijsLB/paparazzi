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
 #define Y_MIN 78
 #define Y_MAX 242
 #define U_MIN 79
 #define U_MAX 169
 #define V_MIN 107
 #define V_MAX 205
 
 // Black & White Filtering Thresholds
 #define X_WHITE 1  // Consecutive white needed to flip run to white
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
 
     // Rotate image 90° clockwise
     uint8_t rotated[width][height * 2]; // [new_height][new_width * 2]
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             int src_idx = (y * width + x) * 2;
             int dst_x = height - 1 - y;
             int dst_y = x;
             rotated[dst_y][dst_x * 2 + 0] = buf[src_idx];     // Y
             rotated[dst_y][dst_x * 2 + 1] = buf[src_idx + 1]; // U (we skip V for rotation)
         }
     }
 
     // Swap width and height
     int temp = width;
     width = height;
     height = temp;
 
     // Point buf to rotated image
     buf = &rotated[0][0];
 
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
 
             if (y >= Y_MIN && y <= Y_MAX && u >= U_MIN && u <= U_MAX && v >= V_MIN && v <= V_MAX) {
                 downsampled[j][i] = 1; // Ground
             } else {
                 downsampled[j][i] = 0; // Obstacle
             }
         }
     }
 
     // Run-flip logic
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
 
     // Extract horizon
     pthread_mutex_lock(&mutex);
     for (int col = 0; col < downsample_w; col++) {
         global_horizon.horizon[col] = downsample_h - 1;
         for (int row = downsample_h - 1; row >= 0; row--) { // Scan from bottom to top
             if (downsampled[row][col] == 1) {
                 global_horizon.horizon[col] = row;
                 break;
             }
         }
     }
     global_horizon.updated = true;
     pthread_mutex_unlock(&mutex);
 
     // Print horizon as 1D array
     printf("Horizon: ");
     for (int i = 0; i < downsample_w; i++) {
         printf("%d ", global_horizon.horizon[i]);
     }
     printf("\n");
 
     // Visualize downsampled image (1 = ground = white block, 0 = obstacle = dark)
     for (int row = 0; row < downsample_h; row++) {
         for (int col = 0; col < downsample_w; col++) {
             printf("%c", downsampled[row][col] ? '#' : '.');
         }
         printf("\n");
     }
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