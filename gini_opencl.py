kernel_arguments={
  'num_samples': 16,
  'num_features': 2000,
  'prime': 37,
  'class_type': 'float'
}

kernel="""
//we are using a hash function to collect frequency counts for classes
//may be inacurate, but the probability for that should be low
#define hash(k) as_int((k)) %% %(prime)d %% %(num_samples)d

//A is a 2d matrix, stored in rows of num_samples x num_features

//a good value for BLOCK_SIZE is probably num_samples
#define SAMPLES %(num_samples)d
#define FEATURES 1
#define LOCAL_MATRIX_SIZE SAMPLES*FEATURES

//A and gini_res have the same dimensions, gini_res will hold the gini scores of all the possible splits
//After this kernel, another one that finds the maximum of all elements in gini_res should be applied
__kernel 
void gini(
  __global float *A, 
  __constant %(class_type)s *sample_classes, 
  __global float *gini_res,
  __global int *debug_params)
{
  __local float A_local[LOCAL_MATRIX_SIZE];
  __local %(class_type)s sample_classes_local[%(num_samples)d];

  unsigned int thread_feature = get_global_id(0);
  unsigned int thread_sample = get_global_id(1);
  debug_params[0]=2>1;//get_global_size(0);
  debug_params[1]=1>2;//get_global_size(1);
  debug_params[2]=%(num_features)d;
  debug_params[3]=%(num_samples)d;

  int main_index=thread_sample*%(num_features)d+thread_feature;

  A_local[thread_sample]=A[main_index];
  barrier(CLK_LOCAL_MEM_FENCE);

  sample_classes_local[thread_sample]=sample_classes[thread_sample];

  barrier(CLK_LOCAL_MEM_FENCE);


  int my_classes_counts[%(num_samples)d];
  int classes_counts[%(num_samples)d*2];

  //initializing classes counts
  for(int i=0;i<%(num_samples)d;i++){
    //classes counts for left samples
    classes_counts[2*i]=0;
    //classes counts for right samples
    classes_counts[2*i+1]=0;
    my_classes_counts[i]=0;  
  }

  //classes counts before the split
  for(int i=0; i<%(num_samples)d; i++){
    int hashed_index=hash(sample_classes_local[i]);
    my_classes_counts[hashed_index]++;
  }

  //debug_params[thread_sample+4]=sample_classes_local[thread_sample];
  debug_params[thread_sample+4]=my_classes_counts[thread_sample];

  barrier(CLK_LOCAL_MEM_FENCE);
  //We now classify the samples acording to threshold
  //We get left and right classes counts after split
  //TODO: probably would be more efficient if A_local would be transposed
  float threshold = A_local[thread_sample];
  float left_total=0.;
  float right_total=0.;
  for(int i=0; i<%(num_samples)d; i++){
    //int index=i*%(num_features)d+thread_feature;
    int hashed_index=hash(sample_classes_local[i]);
    //A_local[index]>threshold returns a 0 or 1, so it increments the index for the right class
    int class=(A_local[i]>threshold);
    right_total=right_total+(float)class;
    left_total=left_total+(1-(float)class);
    classes_counts[2*hashed_index+class]++;
  }
  if(thread_feature==0){
    for(int i=0;i<%(num_samples)d;i++){
      debug_params[i+4+%(num_samples)d]=classes_counts[2*i];
    }
  }
  if(thread_feature==0){
    for(int i=0;i<%(num_samples)d;i++){
      debug_params[i+4+2*%(num_samples)d]=classes_counts[2*i+1];
    }
  }


  //now we have the counts for gini, now we compute it
  float my_gini=1.0;
  float left_gini=1.0;
  float right_gini=1.0;
  float current_class_count;
  float current_class_probability;
  float current_class_probability_squared;
  for(int i=0;i<%(num_samples)d; i++){
    current_class_count=(float)classes_counts[2*i+0];
    current_class_probability=current_class_count/left_total;
    current_class_probability_squared=current_class_probability*current_class_probability;
    left_gini=left_gini-current_class_probability_squared;

    current_class_count=(float)classes_counts[2*i+1];
    current_class_probability=current_class_count/right_total;
    current_class_probability_squared=current_class_probability*current_class_probability;
    right_gini=right_gini-current_class_probability_squared;

    current_class_count=(float)my_classes_counts[i];
    current_class_probability=current_class_count/%(num_samples)f;
    current_class_probability_squared=current_class_probability*current_class_probability;
    my_gini=my_gini-current_class_probability_squared;
  }
   
  float split_score=my_gini-1/%(num_samples)f*(left_total*left_gini+right_total*right_gini);

  gini_res[main_index]=split_score;
  barrier(CLK_GLOBAL_MEM_FENCE);

}"""

#print kernel % kernel_arguments


