#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <stdint.h>
#include <string.h>

#define DOWNSAMPLE_SIZE 10  // Defines the number of divisions for both width and height
#define THRESH_OBSTACLE 128  // Threshold to determine black pixels
#define X_WHITE_TILES 3  // Number of consecutive white tiles needed to flip to free space
#define Y_BLACK_TILES 3  // Number of consecutive black tiles needed to lock obstacle

// YUV Green Filter Ranges
#define Y_MIN 90
#define Y_MAX 210
#define U_MIN 75
#define U_MAX 115
#define V_MIN 69
#define V_MAX 145

void save_jpeg(const char* filename, uint8_t* data, int width, int height, int grayscale) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE* outfile = fopen(filename, "wb");
    if (!outfile) {
        printf("Error: Cannot open output file %s\n", filename);
        return;
    }
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = grayscale ? 1 : 3;
    cinfo.in_color_space = grayscale ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row_pointer = &data[cinfo.next_scanline * width * (grayscale ? 1 : 3)];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}

void apply_yuv_filter(uint8_t* color_data, uint8_t* binary_data, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        uint8_t y = color_data[i * 3];
        uint8_t u = color_data[i * 3 + 1];
        uint8_t v = color_data[i * 3 + 2];
        
        if (y >= Y_MIN && y <= Y_MAX && u >= U_MIN && u <= U_MAX && v >= V_MIN && v <= V_MAX) {
            binary_data[i] = 255; // Mark as white (free space)
        } else {
            binary_data[i] = 0; // Mark as black (obstacle)
        }
    }
}

void apply_black_white_logic(uint8_t* data, int width, int height) {
    for (int row = 0; row < height; row++) {
        int locked = 0;
        int consecutive_black = 0;
        int consecutive_white = 0;
        int black_run_start = -1;
        
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            int is_black = (data[idx] == 0); // Now using the YUV filter output directly

            if (locked) {
                data[idx] = 0; // Once locked, everything remains black
                continue;
            }

            if (is_black) {
                if (black_run_start == -1) {
                    black_run_start = col;
                }
                consecutive_black++;
                consecutive_white = 0;
            } else {
                consecutive_white++;
                
                if (consecutive_white >= X_WHITE_TILES && black_run_start != -1) {
                    for (int i = black_run_start; i < col; i++) {
                        data[row * width + i] = 255; // Flip previous black to white
                    }
                    black_run_start = -1;
                    consecutive_black = 0;
                }
            }

            if (consecutive_black >= Y_BLACK_TILES) {
                locked = 1;
                for (int i = black_run_start; i <= col; i++) {
                    data[row * width + i] = 0; // Lock the rest of the row as black
                }
                black_run_start = -1;
            }
        }
    }
}

int main() {
    char input_path[256];
    char downsampled_path[256];
    char yuv_filtered_path[256];
    char bw_output_path[256];
    snprintf(input_path, sizeof(input_path), "%s/paparazzi/prototyping/decision_logic/image_processing/temp/1204315441.jpg", getenv("HOME"));
    snprintf(downsampled_path, sizeof(downsampled_path), "%s/paparazzi/prototyping/decision_logic/image_processing/temp/downsampled.jpg", getenv("HOME"));
    snprintf(yuv_filtered_path, sizeof(yuv_filtered_path), "%s/paparazzi/prototyping/decision_logic/image_processing/temp/yuv_filtered.jpg", getenv("HOME"));
    snprintf(bw_output_path, sizeof(bw_output_path), "%s/paparazzi/prototyping/decision_logic/image_processing/temp/bw_output.jpg", getenv("HOME"));

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE* infile = fopen(input_path, "rb");
    if (!infile) {
        printf("Error: Cannot open input file %s\n", input_path);
        return -1;
    }
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    int width = cinfo.output_width, height = cinfo.output_height;
    int new_width = width / DOWNSAMPLE_SIZE;
    int new_height = height / DOWNSAMPLE_SIZE;
    uint8_t* image_data = malloc(new_width * new_height * 3);
    uint8_t* binary_data = malloc(new_width * new_height);
    if (!image_data || !binary_data) {
        printf("Error: Memory allocation failed.\n");
        return -1;
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row_pointer = &image_data[(cinfo.output_scanline / DOWNSAMPLE_SIZE) * new_width * 3];
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    save_jpeg(downsampled_path, image_data, new_width, new_height, 0);
    apply_yuv_filter(image_data, binary_data, new_width, new_height);
    save_jpeg(yuv_filtered_path, binary_data, new_width, new_height, 1);
    apply_black_white_logic(binary_data, new_width, new_height);
    save_jpeg(bw_output_path, binary_data, new_width, new_height, 1);
    free(image_data);
    free(binary_data);
    printf("Saved: %s\n", bw_output_path);
    return 0;
}