#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <jpeglib.h>
#include <string.h>
#define Y_MIN 90
#define Y_MAX 210
#define U_MIN 75
#define U_MAX 115
#define V_MIN 69
#define V_MAX 145

// RGB to YUV conversion
void rgb_to_yuv(uint8_t r, uint8_t g, uint8_t b, uint8_t *y, uint8_t *u, uint8_t *v) {
    *y = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
    *u = (uint8_t)(-0.1687 * r - 0.3313 * g + 0.5 * b + 128);
    *v = (uint8_t)(0.5 * r - 0.4187 * g - 0.0813 * b + 128);
}

// Function to filter out stray black and white particles based on vertical neighbors
void filter_particles(uint8_t *image_data, int width, int height) {
    uint8_t *filtered_image = malloc(width * height * 3);  // Temporary image for filtered result
    memcpy(filtered_image, image_data, width * height * 3);  // Copy original image to filtered_image

    // Iterate through each pixel (excluding edges)
    for (int i = 0; i < height; i++) {
        for (int j = 1; j < width - 1; j++) {  // Start from j=1 to avoid edge cases
            int idx = (i * width + j) * 3;

            // Get the current pixel's color (R, G, B values)
            int current_pixel = (image_data[idx] < 128) ? 0 : 1; // Black = 0, White = 1
            int opposite_pixel = (current_pixel == 0) ? 1 : 0;

            // Check the horizontal neighbors (left and right)
            int left_pixel = (image_data[((i * width + j - 1) * 3)] < 128) ? 0 : 1;
            int right_pixel = (image_data[((i * width + j + 1) * 3)] < 128) ? 0 : 1;

            // Count how many opposite neighbors (left and right) there are
            int opposite_count = 0;
            if (left_pixel == opposite_pixel) opposite_count++;
            if (right_pixel == opposite_pixel) opposite_count++;

            // If there are exactly 2 opposite neighbors, change the current pixel
            if (opposite_count == 2) {
                filtered_image[idx] = (opposite_pixel == 0) ? 0 : 255;
                filtered_image[idx + 1] = (opposite_pixel == 0) ? 0 : 255;
                filtered_image[idx + 2] = (opposite_pixel == 0) ? 0 : 255;
            }
        }
    }

    // Copy the filtered result back into the original image
    memcpy(image_data, filtered_image, width * height * 3);

    // Clean up
    free(filtered_image);
}

// Function to process the image, apply YUV filter, downscale, and save the output
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

    // Create a 2D matrix filled with 1s for white pixels and 0s for black pixels
    int **binary_matrix = malloc(height * sizeof(int *));
    for (int i = 0; i < height; i++) {
        binary_matrix[i] = malloc(width * sizeof(int));
    }

    for (int i = 0; i < width * height; i++) {
        uint8_t y, u, v;
        rgb_to_yuv(image_data[i * 3], image_data[i * 3 + 1], image_data[i * 3 + 2], &y, &u, &v);

        // Check if YUV values fall within the defined range for white
        if (y >= Y_MIN && y <= Y_MAX && u >= U_MIN && u <= U_MAX && v >= V_MIN && v <= V_MAX) {
            binary_matrix[i / width][i % width] = 1; // White pixel
        } else {
            binary_matrix[i / width][i % width] = 0; // Black pixel
        }
    }

    // Calculate new dimensions after downscaling (every 5x5 block becomes 1 pixel)
    int new_width = width / 5;
    int new_height = height / 5;

    // Create the downscaled binary image
    uint8_t *downscaled_image = malloc(new_width * new_height * 3);

    // Process each 5x5 block
    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            int white_count = 0;

            // Count the number of white pixels in the 5x5 block
            for (int y = i * 5; y < (i + 1) * 5; y++) {
                for (int x = j * 5; x < (j + 1) * 5; x++) {
                    if (binary_matrix[y][x] == 1) {
                        white_count++;
                    }
                }
            }

            // If the majority of pixels are white, set the output pixel to white, else black
            int idx = (i * new_width + j) * 3;
            if (white_count > 12) {  // More than half of the pixels are white
                downscaled_image[idx] = 255; // R
                downscaled_image[idx + 1] = 255; // G
                downscaled_image[idx + 2] = 255; // B (pure white pixel)
            } else {
                downscaled_image[idx] = 0; // R
                downscaled_image[idx + 1] = 0; // G
                downscaled_image[idx + 2] = 0; // B (pure black pixel)
            }
        }
    }
    // Apply the particle filtering logic to the downscaled image
    filter_particles(downscaled_image, new_width, new_height);

    // Apply rule: after the first black pixel, every pixel after it should be black in that row
    for (int i = 0; i < new_height; i++) {
      int found_black_pixel = 0;
      for (int j = 0; j < new_width; j++) {
          int idx = (i * new_width + j) * 3;

          if (downscaled_image[idx] == 0 && downscaled_image[idx + 1] == 0 && downscaled_image[idx + 2] == 0) {
              found_black_pixel = 1; // Found the first black pixel in this row
          }

          // After the first black pixel, set all subsequent pixels in the row to black
          if (found_black_pixel) {
              downscaled_image[idx] = 0;
              downscaled_image[idx + 1] = 0;
              downscaled_image[idx + 2] = 0;
          }
      }
    }
    
    // Count the number of white pixels in each row, make the last one red, and save the count in an array
    int *white_pixel_counts = malloc(new_height * sizeof(int)); // Array to hold the count of white pixels per row

    for (int i = 0; i < new_height; i++) {
        int white_pixel_count = 0;
        int last_white_pixel_idx = -1;

        // Count white pixels and find the last white pixel index
        for (int j = 0; j < new_width; j++) {
            int idx = (i * new_width + j) * 3;
            
            if (downscaled_image[idx] == 255 && downscaled_image[idx + 1] == 255 && downscaled_image[idx + 2] == 255) {
                white_pixel_count++;  // Count white pixels
                last_white_pixel_idx = j;  // Update the index of the last white pixel
            }
        }

        // Store the count of white pixels for this row
        white_pixel_counts[i] = white_pixel_count;

        // If there is at least one white pixel, make the last one red
        if (last_white_pixel_idx != -1) {
            int idx = (i * new_width + last_white_pixel_idx) * 3;
            downscaled_image[idx] = 255;    // R
            downscaled_image[idx + 1] = 0;  // G
            downscaled_image[idx + 2] = 0;  // B (Red)
        }
    }



    // Save the downscaled image as JPEG
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

    cinfo_out.image_width = new_width;
    cinfo_out.image_height = new_height;
    cinfo_out.input_components = 3;  // RGB
    cinfo_out.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo_out);

    // Set maximum quality (100) for minimal compression
    jpeg_set_quality(&cinfo_out, 100, TRUE);  // Set quality to 100

    // Tighten compression settings for better sharpness (use progressive mode)
    cinfo_out.optimize_coding = TRUE; // Optimize coding for better results
    cinfo_out.dct_method = JDCT_FASTEST; // Use the fastest DCT method

    jpeg_start_compress(&cinfo_out, TRUE);

    uint8_t **downscaled_rows = malloc(sizeof(uint8_t *) * new_height);
    for (int i = 0; i < new_height; i++) {
        downscaled_rows[i] = downscaled_image + i * new_width * 3; // Point to the start of each row
    }

    while (cinfo_out.next_scanline < new_height) {
        jpeg_write_scanlines(&cinfo_out, &downscaled_rows[cinfo_out.next_scanline], 1);
    }

    jpeg_finish_compress(&cinfo_out);
    jpeg_destroy_compress(&cinfo_out);
    fclose(output_file);

    // Clean up memory
    free(downscaled_image);
    for (int i = 0; i < height; i++) {
        free(binary_matrix[i]);
    }
    free(binary_matrix);

}

// Main function to run the program
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    process_image(argv[1], argv[2]);

    return 0;
}
