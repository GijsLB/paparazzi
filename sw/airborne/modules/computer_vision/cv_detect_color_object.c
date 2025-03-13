/*
 * Copyright (C) 2019 Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of Paparazzi.
 *
 * Paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * Paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * @file modules/computer_vision/cv_detect_object.h
 * Assumes the object consists of a continuous color and checks
 * if you are over the defined object or not
 */

// Own header
#include "modules/computer_vision/cv_detect_color_object.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"
//#include <opencv/cv.h>
//#include <opencv/highgui.h>

#define PRINT(string,...) fprintf(stderr, "[object_detector->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if OBJECT_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static pthread_mutex_t mutex;

#ifndef COLOR_OBJECT_DETECTOR_FPS1
#define COLOR_OBJECT_DETECTOR_FPS1 0 ///< Default FPS (zero means run at camera fps)
#endif
#ifndef COLOR_OBJECT_DETECTOR_FPS2
#define COLOR_OBJECT_DETECTOR_FPS2 0 ///< Default FPS (zero means run at camera fps)
#endif

// Filter Settings
uint8_t cod_lum_min1 = 0;
uint8_t cod_lum_max1 = 0;
uint8_t cod_cb_min1 = 0;
uint8_t cod_cb_max1 = 0;
uint8_t cod_cr_min1 = 0;
uint8_t cod_cr_max1 = 0;

uint8_t cod_lum_min2 = 0;
uint8_t cod_lum_max2 = 0;
uint8_t cod_cb_min2 = 0;
uint8_t cod_cb_max2 = 0;
uint8_t cod_cr_min2 = 0;
uint8_t cod_cr_max2 = 0;

bool cod_draw1 = false;
bool cod_draw2 = false;

// define global variables
struct color_object_t {
  int32_t x_c;
  int32_t y_c;
  uint32_t color_count;
  bool updated;
};
struct color_object_t global_filters[2];

struct image_data_t {
  // struct image_t *img;
  bool updated;
  uint32_t edge_count;
};
static struct image_data_t global_image;

// Declare function
uint32_t get_edge_score(struct image_t *img);


// Function
uint32_t find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max);

/*
 * object_detector
 * @param img - input image to process
 * @param filter - which detection filter to process
 * @return img
 */
static struct image_t *object_detector(struct image_t *img, uint8_t filter)
{
  uint8_t lum_min, lum_max;
  uint8_t cb_min, cb_max;
  uint8_t cr_min, cr_max;
  bool draw;

  switch (filter){
    case 1:
      lum_min = cod_lum_min1;
      lum_max = cod_lum_max1;
      cb_min = cod_cb_min1;
      cb_max = cod_cb_max1;
      cr_min = cod_cr_min1;
      cr_max = cod_cr_max1;
      draw = cod_draw1;
      break;
    case 2:
      lum_min = cod_lum_min2;
      lum_max = cod_lum_max2;
      cb_min = cod_cb_min2;
      cb_max = cod_cb_max2;
      cr_min = cod_cr_min2;
      cr_max = cod_cr_max2;
      draw = cod_draw2;
      break;
    default:
      return img;
  };

  int32_t x_c, y_c;

  // Filter and find centroid
  uint32_t count = find_object_centroid(img, &x_c, &y_c, draw, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);
  VERBOSE_PRINT("Color count %d: %u, threshold %u, x_c %d, y_c %d\n", camera, object_count, count_threshold, x_c, y_c);
  VERBOSE_PRINT("centroid %d: (%d, %d) r: %4.2f a: %4.2f\n", camera, x_c, y_c,
        hypotf(x_c, y_c) / hypotf(img->w * 0.5, img->h * 0.5), RadOfDeg(atan2f(y_c, x_c)));

  pthread_mutex_lock(&mutex);
  // Update colors
  global_filters[filter-1].color_count = count;
  global_filters[filter-1].x_c = x_c;
  global_filters[filter-1].y_c = y_c;
  global_filters[filter-1].updated = true;

  // Update global image, which is to be used by edge detection
  global_image.edge_count = get_edge_score(img);
  global_image.updated = true;
  printf("GLOBAL IMAGE UPDATED!");
  pthread_mutex_unlock(&mutex);

  return img;
}

struct image_t *object_detector1(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 1);
}

struct image_t *object_detector2(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector2(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 2);
}

void color_object_detector_init(void)
{
  memset(global_filters, 0, 2*sizeof(struct color_object_t));
  pthread_mutex_init(&mutex, NULL);

  // Initialize global_image
  // global_image.img = NULL;  
  global_image.edge_count = 0;
  global_image.updated = false;

#ifdef COLOR_OBJECT_DETECTOR_CAMERA1
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN1
  cod_lum_min1 = COLOR_OBJECT_DETECTOR_LUM_MIN1;
  cod_lum_max1 = COLOR_OBJECT_DETECTOR_LUM_MAX1;
  cod_cb_min1 = COLOR_OBJECT_DETECTOR_CB_MIN1;
  cod_cb_max1 = COLOR_OBJECT_DETECTOR_CB_MAX1;
  cod_cr_min1 = COLOR_OBJECT_DETECTOR_CR_MIN1;
  cod_cr_max1 = COLOR_OBJECT_DETECTOR_CR_MAX1;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW1
  cod_draw1 = COLOR_OBJECT_DETECTOR_DRAW1;
#endif

  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, object_detector1, COLOR_OBJECT_DETECTOR_FPS1, 0);
#endif

#ifdef COLOR_OBJECT_DETECTOR_CAMERA2
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN2
  cod_lum_min2 = COLOR_OBJECT_DETECTOR_LUM_MIN2;
  cod_lum_max2 = COLOR_OBJECT_DETECTOR_LUM_MAX2;
  cod_cb_min2 = COLOR_OBJECT_DETECTOR_CB_MIN2;
  cod_cb_max2 = COLOR_OBJECT_DETECTOR_CB_MAX2;
  cod_cr_min2 = COLOR_OBJECT_DETECTOR_CR_MIN2;
  cod_cr_max2 = COLOR_OBJECT_DETECTOR_CR_MAX2;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW2
  cod_draw2 = COLOR_OBJECT_DETECTOR_DRAW2;
#endif

  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA2, object_detector2, COLOR_OBJECT_DETECTOR_FPS2, 1);
#endif
}

/*
 * find_object_centroid
 *
 * Finds the centroid of pixels in an image within filter bounds.
 * Also returns the amount of pixels that satisfy these filter bounds.
 *
 * @param img - input image to process formatted as YUV422.
 * @param p_xc - x coordinate of the centroid of color object
 * @param p_yc - y coordinate of the centroid of color object
 * @param lum_min - minimum y value for the filter in YCbCr colorspace
 * @param lum_max - maximum y value for the filter in YCbCr colorspace
 * @param cb_min - minimum cb value for the filter in YCbCr colorspace
 * @param cb_max - maximum cb value for the filter in YCbCr colorspace
 * @param cr_min - minimum cr value for the filter in YCbCr colorspace
 * @param cr_max - maximum cr value for the filter in YCbCr colorspace
 * @param draw - whether or not to draw on image
 * @return number of pixels of image within the filter bounds.
 */
uint32_t find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max)
{
  uint32_t cnt = 0;
  uint32_t tot_x = 0;
  uint32_t tot_y = 0;
  uint8_t *buffer = img->buf;

  // printf("Does find_object_centroid work? img->w: %d\n", img->w);

  // Go through all the pixels
  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x ++) {
      // Check if the color is inside the specified values
      uint8_t *yp, *up, *vp;
      if (x % 2 == 0) {
        // Even x
        up = &buffer[y * 2 * img->w + 2 * x];      // U
        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y1
        vp = &buffer[y * 2 * img->w + 2 * x + 2];  // V
        //yp = &buffer[y * 2 * img->w + 2 * x + 3]; // Y2
      } else {
        // Uneven x
        up = &buffer[y * 2 * img->w + 2 * x - 2];  // U
        //yp = &buffer[y * 2 * img->w + 2 * x - 1]; // Y1
        vp = &buffer[y * 2 * img->w + 2 * x];      // V
        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y2
      }
      if ( (*yp >= lum_min) && (*yp <= lum_max) &&
           (*up >= cb_min ) && (*up <= cb_max ) &&
           (*vp >= cr_min ) && (*vp <= cr_max )) {
        cnt ++;
        tot_x += x;
        tot_y += y;
        if (draw){
          *yp = 255;  // make pixel brighter in image
        }
      }
    }
  }
  if (cnt > 0) {
    *p_xc = (int32_t)roundf(tot_x / ((float) cnt) - img->w * 0.5f);
    *p_yc = (int32_t)roundf(img->h * 0.5f - tot_y / ((float) cnt));
  } else {
    *p_xc = 0;
    *p_yc = 0;
  }
  return cnt;
}

/*
 * get_edge_score
 * 
 * Function that finds the edge score
 */
uint32_t get_edge_score(struct image_t *img)
{
  printf("get_edge_score is called\n");
  if (img != NULL) {
    printf("We got an image, lets go! Size w x h: %d x %d\n", img->w, img->h);
  } else {
    return 0;
  }

  uint32_t edge_count = 0;      // Stores the number of detected edges
  uint8_t *buffer = img->buf;
  uint8_t width = img->w;
  uint8_t height = img->h;
  uint8_t row_stride = 2 * width;
  // uint32_t edge_arr[height - 2][width - 2];   // Define the edge array 


  // printf("break1");

  // Define the kernels
  int8_t G_x[3][3] = {{-1, 0, 1},
                        {-2, 0, 2},
                        {-1, 0, 1}}; 
  int8_t G_y[3][3] = {{-1, -2, -1},
                        {0, 0, 0},
                        {1, 2, 1}}; 

  // Loop through each pixel, skipping the first and last rows/columns
  for (uint16_t y = 1; y < height - 1; y++) {
      for (uint16_t x = 1; x < width - 1; x++) {
        // Extract the surrounding 8 pixels (Y values)
        uint8_t P_TL  = buffer[(y - 1) * row_stride + 2 * (x - 1) + 1]; // Top-left
        uint8_t P_T   = buffer[(y - 1) * row_stride + 2 * x + 1];       // Top
        uint8_t P_TR  = buffer[(y - 1) * row_stride + 2 * (x + 1) + 1]; // Top-right
        uint8_t P_L   = buffer[y * row_stride + 2 * (x - 1) + 1];       // Left
        uint8_t P_C   = buffer[y * row_stride + 2 * x + 1];             // Center
        uint8_t P_R   = buffer[y * row_stride + 2 * (x + 1) + 1];       // Right
        uint8_t P_BL  = buffer[(y + 1) * row_stride + 2 * (x - 1) + 1]; // Bottom-left
        uint8_t P_B   = buffer[(y + 1) * row_stride + 2 * x + 1];       // Bottom
        uint8_t P_BR  = buffer[(y + 1) * row_stride + 2 * (x + 1) + 1]; // Bottom-right

        // Compute G_x
        int32_t Gx = (G_x[0][0] * P_TL) + (G_x[0][1] * P_T) + (G_x[0][2] * P_TR) +
                     (G_x[1][0] * P_L)  + (G_x[1][1] * P_C) + (G_x[1][2] * P_R) +
                     (G_x[2][0] * P_BL) + (G_x[2][1] * P_B) + (G_x[2][2] * P_BR);

        // Compute G_y
        int32_t Gy = (G_y[0][0] * P_TL) + (G_y[0][1] * P_T) + (G_y[0][2] * P_TR) +
                     (G_y[1][0] * P_L)  + (G_y[1][1] * P_C) + (G_y[1][2] * P_R) +
                     (G_y[2][0] * P_BL) + (G_y[2][1] * P_B) + (G_y[2][2] * P_BR);

        // Compute Gradient Magnitude: G = sqrt(Gx^2 + Gy^2)
        uint32_t G = sqrt(Gx * Gx + Gy * Gy);

        // Store result in edge array
        // edge_arr[y - 1][x - 1] = G;

        // printf("break x: %d\n", x);
        // printf("break y: %d\n", y);


        // Thresholding: If G is strong enough, count it as an edge
        if (G > 100) {
          edge_count++;
        }
      }
    }
    printf("EDGE_COUNT CV_DETECT!! %d\n", edge_count);
    return edge_count;
} 

void color_object_detector_periodic(void)
{
  static struct color_object_t local_filters[2];
  // static struct image_t *current_image = NULL;  // Declare before using it

  pthread_mutex_lock(&mutex);
  memcpy(local_filters, global_filters, 2*sizeof(struct color_object_t));

  // // Get last image safely
  // if (global_image.updated) {
  //   current_image = global_image.img;
  //   global_image.updated = false;
  // }
  pthread_mutex_unlock(&mutex);

  if(local_filters[0].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, local_filters[0].x_c, local_filters[0].y_c,
        0, 0, local_filters[0].color_count, 0);
    local_filters[0].updated = false;
  }
  if(local_filters[1].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION2_ID, local_filters[1].x_c, local_filters[1].y_c,
        0, 0, local_filters[1].color_count, 1);
    local_filters[1].updated = false;
  }

  if (global_image.updated) {
    uint32_t edge_count = global_image.edge_count;
    printf("SEND THE MESSAGE!");
    AbiSendMsgEDGE_COUNT(EDGE_COUNT_ID, edge_count);
    global_image.updated = false;
  }

  // if (current_image != NULL) {
  //   uint32_t edge_count = get_edge_score(current_image);
  //   printf("SEND THE MESSAGE!");
  //   AbiSendMsgEDGE_COUNT(EDGE_COUNT_ID, edge_count);
  //   global_image.updated = false;
  // } else {
  //     printf("[color_object_detector_periodic] Warning: current_image is NULL!\n");
  // }
}
