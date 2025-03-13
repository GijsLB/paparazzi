#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    const char *image_name = "1231615251.jpg";  // Last assigned filename
    const char *input_dir = "~/paparazzi/prototyping/collected_datasets/Test3_7maart_tapijt";

    // Expand `~` to home directory
    const char *home = getenv("HOME");
    char full_path[512];

    if (input_dir[0] == '~' && home) {
        snprintf(full_path, sizeof(full_path), "%s%s/%s", home, input_dir + 1, image_name);
    } else {
        snprintf(full_path, sizeof(full_path), "%s/%s", input_dir, image_name);
    }

    printf("Image path: %s\n", full_path);

    return 0;
}

uint32_t find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
  uint8_t lum_min, uint8_t lum_max,
  uint8_t cb_min, uint8_t cb_max,
  uint8_t cr_min, uint8_t cr_max)
{
uint32_t cnt = 0;
uint32_t tot_x = 0;
uint32_t tot_y = 0;
uint8_t *buffer = img->buf;

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