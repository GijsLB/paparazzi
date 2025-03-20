#ifndef OPTICALFLOW_AVOIDER_H
#define OPTICALFLOW_AVOIDER_H

// External variable declaration
extern float obstacle_threshold;

void opticalflow_avoider_init(void);
void opticalflow_avoider_periodic(void);

#endif /* OPTICALFLOW_AVOIDER_H */
