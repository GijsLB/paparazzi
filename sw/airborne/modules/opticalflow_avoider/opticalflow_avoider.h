#ifndef OPTICALFLOW_AVOIDER_H
#define OPTICALFLOW_AVOIDER_H

// Externe variabele declaratie
extern float obstacle_threshold;
extern float OF_THRESHOLD;


void opticalflow_avoider_init(void);
void opticalflow_avoider_periodic(void);

#endif /* OPTICALFLOW_AVOIDER_H */
