import os
import glob

from utils import scoring_utils
from utils import plotting_tools 
from utils import model_tools

### Test on Pics ####
list_of_files = glob.glob('../data/weights/*') # * means all if need specific format then *.csv
# Last created file
weight_file_name =  max(list_of_files, key=os.path.getctime)
weight_file_name = weight_file_name.split("\\")

if 'config_' in weight_file_name[-1]:
    model = model_tools.load_network(weight_file_name[-2])
else:
    model = model_tools.load_network(weight_file_name[-1])


#for iii in range(len(weight_file_name)):
#    if not 'config_' in weight_file_name[iii]:
#        weight_file_nam = weight_file_name[iii].split("\\")
#        model_tools.load_network(weight_file_nam[-1])
        
        
run_num = 'run_1'


val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_with_targ', 'sample_evaluation_data') 

val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                        run_num,'patrol_non_targ', 'sample_evaluation_data') 

val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                        run_num,'following_images', 'sample_evaluation_data')


print("Images while following the target")
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','following_images', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
    
    
print("Images while at patrol without target")
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_non_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
 
    
print("Images while at patrol with target")
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_with_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)



#### Evaluate ####

print("Quad behind the target")
true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)

print("Target not visible")
true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)

print("Target far away")
true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)

# Sum all the true positives, etc from the three datasets to get a weight for the score
true_pos = true_pos1 + true_pos2 + true_pos3
false_pos = false_pos1 + false_pos2 + false_pos3
false_neg = false_neg1 + false_neg2 + false_neg3

weight = true_pos/(true_pos+false_neg+false_pos)
#print(weight)


# The IoU for the dataset that never includes the hero is excluded from grading
final_IoU = (iou1 + iou3)/2
print("IoU no hero - ", final_IoU)


# And the final grade score is 
final_score = final_IoU * weight
print("Final Grade - ", final_score)
        
        #if fs < final_score:
        #    fs=final_score
        #    nam = weight_file_name[iii]
        
            
#print('score',fs)
#print('nam',nam)