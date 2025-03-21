#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <stdint.h>

// Define YUV range for green detection
#define Y_MIN 90
#define Y_MAX 210
#define U_MIN 75
#define U_MAX 115
#define V_MIN 69
#define V_MAX 145

// Define tunable parameters for filtering ground and non-ground areas
#define Y_BLACK 8 // Consecutive black pixels before turning white pixels to black
#define X_WHITE 10 // Consecutive white pixels before turning black pixels to white

void rgb_to_yuv(uint8_t r, uint8_t g, uint8_t b, uint8_t *y, uint8_t *u, uint8_t *v) {
    *y = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
    *u = (uint8_t)(-0.1687 * r - 0.3313 * g + 0.5 * b + 128);
    *v = (uint8_t)(0.5 * r - 0.4187 * g - 0.0813 * b + 128);
}

void filter_ground(uint8_t *image_data, int width, int height, int *horizon) {
  for (int x = 0; x < height; x++) {  // Loop over rotated x-axis
      int black_count = 0;  
      int white_count = 0;  
      int ground_start = -1;
      int wall_start = -1;
      int highest_white = -1; // Store highest detected white pixel
      int highest_white_y = -1; // Store the y-coordinate of the highest white pixel

      for (int y = 0; y < width; y++) {  // Scan **top to bottom**
          int idx = (y * height + x) * 3;  // Corrected indexing

          uint8_t r = image_data[idx];
          uint8_t g = image_data[idx + 1];
          uint8_t b = image_data[idx + 2];

          int is_black = (r == 0 && g == 0 && b == 0);
          int is_white = (r == 255 && g == 255 && b == 255);

          if (is_black) {
              black_count++;
              white_count = 0;

              if (black_count >= Y_BLACK && wall_start == -1) {
                  wall_start = y;
              }
          } else if (is_white) {
              white_count++;
              black_count = 0;

              if (white_count >= X_WHITE && ground_start == -1) {
                  ground_start = y;
              }

              // Update the highest white pixel (smallest y value)
              if (highest_white == -1 || y < highest_white) {
                  highest_white = y;
                  highest_white_y = y;  // Store the y-coordinate of the highest white pixel
              }
          }
      }

      // Debugging output for tracking
      // printf("Column %d: highest_white = %d, ground_start = %d, wall_start = %d\n", x, highest_white, ground_start, wall_start);

      // If no white pixel found, set horizon to the middle of the image as a fallback
      horizon[x] = (highest_white == -1) ? height / 2 : highest_white;

      // Turn everything below ground_start to white
      if (ground_start != -1) {
          for (int y = ground_start; y < width; y++) {
              int idx = (y * height + x) * 3;  // Corrected indexing
              image_data[idx] = 255;
              image_data[idx + 1] = 255;
              image_data[idx + 2] = 255;
          }
      }

      // Turn everything above wall_start to black
      if (wall_start != -1) {
          for (int y = wall_start; y >= 0; y--) {
              int idx = (y * height + x) * 3;  // Corrected indexing
              image_data[idx] = 0;
              image_data[idx + 1] = 0;
              image_data[idx + 2] = 0;
          }
      }

      // Mark the highest white pixel as red
      // if (highest_white_y != -1) {
      //   int idx = (highest_white_y * height + x) * 3;
      //   image_data[idx] = 255;    // Red
      //   image_data[idx + 1] = 0;  // Green
      //   image_data[idx + 2] = 0;  // Blue
      // }
  }

  // Save the horizon data to a file
  FILE *horizon_file = fopen("horizon.txt", "w");
  if (horizon_file) {
      for (int x = 0; x < height; x++) {
          fprintf(horizon_file, "%d\n", horizon[x]);
      }
      fclose(horizon_file);
  }
}


void process_image(const char *input_filename, const char *output_filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *input_file = fopen(input_filename, "rb");
    if (!input_file) {
        perror("Error opening input file");
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, input_file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int pixel_size = cinfo.output_components;

    uint8_t *image_data = malloc(width * height * pixel_size);
    uint8_t **row_pointers = malloc(sizeof(uint8_t *) * height);
    for (int i = 0; i < height; i++) {
        row_pointers[i] = image_data + i * width * pixel_size;
    }

    while (cinfo.output_scanline < height) {
        jpeg_read_scanlines(&cinfo, &row_pointers[cinfo.output_scanline], 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(input_file);

    // Rotate 90 degrees counterclockwise
    uint8_t *rotated_data = malloc(width * height * pixel_size);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * pixel_size;
            int dst_idx = ((width - x - 1) * height + y) * pixel_size;
            for (int c = 0; c < pixel_size; c++) {
                rotated_data[dst_idx + c] = image_data[src_idx + c];
            }
        }
    }
    free(image_data);

    // Apply YUV422 filter
    for (int i = 0; i < width * height; i++) {
        uint8_t y, u, v;
        rgb_to_yuv(rotated_data[i * 3], rotated_data[i * 3 + 1], rotated_data[i * 3 + 2], &y, &u, &v);
        printf("Pixel %d -> Y: %d, U: %d, V: %d\n", i, y, u, v);
        if (y >= Y_MIN && y <= Y_MAX && u >= U_MIN && u <= U_MAX && v >= V_MIN && v <= V_MAX) {
            rotated_data[i * 3] = 255;
            rotated_data[i * 3 + 1] = 255;
            rotated_data[i * 3 + 2] = 255;
        } else {
            rotated_data[i * 3] = 0;
            rotated_data[i * 3 + 1] = 0;
            rotated_data[i * 3 + 2] = 0;
        }
    }

    // Store horizon values
    int *horizon = malloc(height * sizeof(int));
    filter_ground(rotated_data, width, height, horizon);
    free(horizon);

    // Save output image
    struct jpeg_compress_struct cinfo_out;
    struct jpeg_error_mgr jerr_out;
    FILE *output_file = fopen(output_filename, "wb");
    if (!output_file) {
        perror("Error opening output file");
        return;
    }

    cinfo_out.err = jpeg_std_error(&jerr_out);
    jpeg_create_compress(&cinfo_out);
    jpeg_stdio_dest(&cinfo_out, output_file);

    cinfo_out.image_width = height;
    cinfo_out.image_height = width;
    cinfo_out.input_components = 3;
    cinfo_out.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo_out);
    jpeg_set_quality(&cinfo_out, 90, TRUE);
    jpeg_start_compress(&cinfo_out, TRUE);

    uint8_t **rotated_rows = malloc(sizeof(uint8_t *) * width);
    for (int i = 0; i < width; i++) {
        rotated_rows[i] = rotated_data + i * height * pixel_size;
    }

    while (cinfo_out.next_scanline < width) {
        jpeg_write_scanlines(&cinfo_out, &rotated_rows[cinfo_out.next_scanline], 1);
    }

    jpeg_finish_compress(&cinfo_out);
    jpeg_destroy_compress(&cinfo_out);
    fclose(output_file);

    free(rotated_data);
    free(rotated_rows);
}

int main() {
    process_image("test2.jpg", "output.jpg");
    return 0;
}