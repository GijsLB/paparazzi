/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.h"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 */

 #ifndef BLACKSEEK_H
 #define BLACKSEEK_H
 
 // settings
 extern float bs_color_count_frac;
 
 // functions
 extern void blackseek_init(void);
 extern void blackseek_periodic(void);
 
 #endif
 
 